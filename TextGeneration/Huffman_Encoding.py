#Huffman Encoding
#Tree-Node Type
class Node:
    def __init__(self,weight):
        self.left = None
        self.right = None
        self.father = None
        self.weight = weight
    def isLeft(self):
        return self.father.left == self
#create nodes
def createNodes(weights):
    return [Node(weight) for weight in weights]

#create Huffman Tree
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:float(item.weight))
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.weight + node_right.weight)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]
#Huffman encoding
def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes