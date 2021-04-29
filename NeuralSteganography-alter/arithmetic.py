import torch
import torch.nn.functional as F

from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg

def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cpu', temp=1.0, precision=16, topk=50000):

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)#隐写文本所用语境（将索引变为张量）

    max_val = 2**precision
    threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context#前缀初始化为语境
    output = context#输出序列先把语境放入，用于在其之后生成隐写文本
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0
    total_num_sents = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)#输出分别为每个单词概率与隐藏层参数
            past = limit_past(past)
            logits[0, -1, -1] = -1e20 # endoftext token can't happen 字典中的“endoftext”符号不会出现，概率设为极低
            logits[0, -1, 628] = -1e20 # 2 newlines token can't happen 字典中的“换两行”符号不会出现，概率设为极低
            logits, indices = logits[0, -1, :].sort(descending=True)#对各概率进行降序排序,输出降序排列的概率及其对应的索引
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)
            
            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0#核心变量，选取概率的索引
                sent_finish = is_sent_finish(indices[selection].item(), enc)#如果已经结尾，循环结束
            else:
                #获取下一单词的索引

                # Cutoff low probabilities that would be rounded to 0 设置概率阈值防止概率太低
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k] # Cutoff all but top k 只留概率前k高的各位

                # Rescale to correct range 调整概率=原概率/概率和*位数表示值的范围（此模型的范围为[0,2^16=65536)，16为隐藏比特流位数（两个字节）)
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range
                

                # Round probabilities to integers given precision 将概率调整为足够大的长整数
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)#按张量的行计算每一个位置的元素（概率）与之前所有元素（概率）的和，并作为此位置的新值————最后一个元素代表总概率和！

                # Remove any elements from the bottom if rounding caused the total prob to be too large 防止概率和过大
                overfill_index = (cum_probs > cur_int_range).nonzero()#提取所有超出概率和上限的概率和的索引
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]#将所有超出上限的概率和缩小到范围内

                # Add any mass to the top if removing/rounding causes the total prob to be too small 防止概率和过小
                cum_probs += cur_int_range-cum_probs[-1] #概率和若过小，加上概率范围与概率和的差值
                
                # Get out resulting probabilities 获取最终概率分布
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]#后一元素与前一元素相减，代表实际的概率分布区域
                
                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i+precision]#获取当前需要隐藏的信息比特流（共precision位）
                if i+precision > len(message):
                    message_bits = message_bits + [0]*(i+precision-len(message))#若此时隐写文本加上新隐写信息后过长，此处逆序信息编码后面的空缺部分补0
                message_idx = bits2int(reversed(message_bits))#将逆序存储的比特流转换为单词索引
                selection = (cum_probs > message_idx).nonzero()[0].item()#选择概率最大且在概率范围内的单词索引（作为下一个词）


                #概率范围更新
                # Calculate new range as ints 计算新的概率上下限
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits 将新的上下限值转换为二进制
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval 计算上下限的二进制值开始不同的地方（新单词的概率二进制值前缀就等于阈值开头到这里）
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded
                #将上限、下限不同之处之后的位均设为1和0（保证选择的概率二进制值不会超出范围）
                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded
                #将上下限重新转换为整数
                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

                # Gather statistics 获取其他属性数值
                total_log_probs += log_probs[selection].item()

                q = probs_final.double()/probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1
            
            # Update history with new token
            prev = indices[selection].view(1)#根据索引selection获取新的单词
            output = torch.cat((output, prev))#将单词加入输出序列中，最后输出作为语境的前文之后的生成文本
            total_num += 1
            #print(enc.decode(prev.tolist()), message_bits[:num_bits_encoded])
            
            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break
            
    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq

def decode_arithmetic(model, enc, text, context, device='cpu', temp=1.0, precision=16, topk=50000):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)#将提供的隐写文本进行编码
    # common BPE error case: 198, 198 (1 newlines) is interpretted as 628 (2 newlines) #198代表换1行，628代表换2行
    i = 0
    while i < len(inp):#纠正BPE的解释错误：把换两行当成换一行
        if inp[i] == 628:
            inp[i] = 198
            inp[i+1:i+1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2**precision
    threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10 # endoftext can't happen
            logits[0, -1, 628] = -1e10 # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            
            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1]-cur_interval[0]
            cur_threshold = 1/cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k] # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()#获取排序后的概率标签与实际的隐写文本中的文本标签中相同部分
            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break
                    
                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix) # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                              true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i+num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i+j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix) # a list
                                inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0
            
            selection = rank#填入的下一个词就是相同部分！
            
            # Calculate new range as ints
            new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
            
            # Emit most significant bits which are now fixed and update interval 找出概率范围代表的单词索引，加入隐写文本，并更新概率范围
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)#寻找二进制流中的相同前缀（在本模型中代表一个单词条件概率的范围开头)
            if i == len(inp)-1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive
            
            # Update history with new token 更新此位置上一位的字符
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            #print(enc.decode([inp[i]]), new_bits)
            i += 1
    
    return message

