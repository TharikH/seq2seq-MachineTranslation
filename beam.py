import heapq
import torch

class BeamNode(object):
    '''
    Beam search uses this class's object as nodes to store all relevent information
    '''
    def __init__(self, probability = 1, path_probability = 0, index = 1, hidden = None, cell = None, parent = None):
        # Probability of this character for predicted
        self.probability = probability
        # Total probability of the path
        self.path_probability = path_probability
        # index to store, so can be passed as input
        self.index = index
        # store the next element
        self.parent = parent
        # To store the prev hidden
        self.hidden = hidden
        # To store previous cell in case of lstm
        self.cell = cell
        # To store path length
        self.length = 0
    
    def __lt__(self, other):
        # As heapq works on min heap, override less than in a way such that highest pah prob will be taken
        return self.path_probability > other.path_probability


class BeamSearch():
    '''
    Use beam search heuristics to decode. Here it is similar to bfs, but we only choose top k nodes to proceed
    where k is the beam width.
    '''
    def __init__(self, beam_width = 3):
        
        # Beam width
        self.beam_width = beam_width 
        
        # To store the nodes to be explored
        self.open_list = []
        
        # make it into heap
        heapq.heapify(self.open_list)
        
        # To store all (based on beam width) probable paths
        self.paths = []
        
        
    def beamSearch(self, model, outputs,dec_hiddens,cells, predicted):
        batch_size = outputs.shape[1]
        
        for i in range(batch_size):
            with torch.no_grad():
                model.eval()
                output = outputs[:,i:i+1].contiguous()
                index = output.contiguous()
                dec_hidden = dec_hiddens[:,i:i+1,:].contiguous()
                cell = cells[:,i:i+1,:].contiguous() if cells is not None else None
                
                # create root node => which has 1 (start index) as the element
                node = BeamNode(1,1,index,dec_hidden, cell, None)
                heapq.heappush(self.open_list,node)

                # bfs loop
                while(len(self.open_list) > 0):
                    curr_node = heapq.heappop(self.open_list)
                    
                    if curr_node.length == model.output_seq_length-1:
                        self.paths.append(curr_node)
                        continue

                    output,dec_hidden,cell,attention_weights=model.decoder.forward(curr_node.index,curr_node.hidden,curr_node.cell,None)
                    output = model.softmax(output,dim=2)

                    # take top k  elements from the output
                    topk, topk_index = torch.topk(output,self.beam_width, dim = 2)

                    # push every neighbour of the curr_node ( bfs neighbours)
                    for j in range(self.beam_width):
                        output = topk[:,:,j]
                        index = topk_index[:,:,j]
                        
                        # If prob is less than some threshold, then stop progressing ( to ensure performance )
                        if curr_node.path_probability * output.item() < 0.001:
                            continue
                        node = BeamNode(output.item(),curr_node.path_probability * output.item(),index,dec_hidden, cell, curr_node)
                        node.length = curr_node.length+1
                        heapq.heappush(self.open_list,node)


                    # Take only k elements to queue in total instead of all nodes, here based on top probabilities
                    self.open_list = heapq.nsmallest(self.beam_width, self.open_list)

                # out of all the paths explored, take the largest probability path
                
            if len(self.paths) > 0:
                path = min(self.paths)
                self.paths = []

                # path will be in reversed order, so reversing to make path correct
                prev = None
                current = path
                while(current is not None):
                    next = current.parent
                    current.parent = prev
                    prev = current
                    current = next
                path = prev

    #             model.train()
                # traverse the path according to the path
                for t in range(1,model.output_seq_length):
                    output,dec_hidden,cell,attention_weights=model.decoder.forward(path.index,path.hidden,path.cell,None)

                    predicted[t,i:i+1] = output

                    path = path.parent
            else:
                output = outputs[:,i:i+1].contiguous()
                index = output.contiguous()
                dec_hidden = dec_hiddens[:,i:i+1,:].contiguous()
                for t in range(1,model.output_seq_length):
                    output,dec_hidden,cell,attention_weights=model.decoder.forward(index,dec_hidden,cell,None)
                    predicted[t,i:i+1] = output
                    output = model.softmax(output,dim=2)
                    output = torch.argmax(output,dim=2)
                    
                