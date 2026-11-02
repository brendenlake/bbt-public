import torch
import numpy as np
from train_utils import extract, display_input_output

def display_sys_gen_html_unmap(samples_pred, fid, freq='percent', include_support=False):
    #  Show model predictions when sampling. 
    #   Each episode consists of just one command, repeated multiple times. 
    #   There must also be a remapping back to canonical text format.
    #
    #   Input:
    #     samples_pred : list of dicts from evaluate_acc, which has predictions for each episode
    #     fid : handle for text file we are writing to
    #     freq : [percent OR count], format for reporting frequency
    #     include_support : show support set in HTML format? used for probe task
    fid.write('var all_data = [')
    for idx,sample in enumerate(samples_pred):
        assert freq in ['percent','count']
        assert('unmap_input' in sample['aux']), "there must be mapping back to canonical text form"
        ui = lambda x : list(map(sample['aux']['unmap_input'], x))
        uo = lambda x : list(map(sample['aux']['unmap_output'], x))
        in_support = sample['in_support']
        xq_novel = extract(np.logical_not(in_support), ui(sample['xq']))
        yq_predict_novel = extract(np.logical_not(in_support),uo(sample['yq_predict']))
        xq_novel = [' '.join(x) for x in xq_novel]
        assert all(xq==xq_novel[0] for xq in xq_novel), "each episode must be repeats of the same command"
        mycommand = xq_novel[0]
        myresponses = [' '.join(y) for y in yq_predict_novel]
        unique_responses = sorted(set(myresponses))
        count_unique = []
        for u in unique_responses:
            if freq=='percent':
                count_unique.append(100.*np.mean(np.array([u == rr for rr in myresponses],dtype=float)))
            else:
                count_unique.append(np.sum(np.array([u == rr for rr in myresponses],dtype=int)))
        unique_responses = [x for x in sorted(zip(unique_responses,count_unique), key=lambda pair: pair[1], reverse=True)]
        if include_support:
            fid.write('[ \n')
            fid.write("'support', \n")
            xs = ui(sample['xs'])
            ys = uo(sample['ys'])
            for j in range(len(xs)):
                fid.write("  ['%s', '%s'], \n" % (' '.join(xs[j]), ' '.join(ys[j]) ))
            if 'output_pool' in sample['aux']:
                mypool = sample['aux']['output_pool']
                fid.write("  ['%s', '%s'], \n" % ('Pool', ' '.join(uo([mypool])[0])) )
            fid.write('], \n')
        fid.write('[ \n')
        if include_support: fid.write("'query', \n")
        for u in unique_responses:
            if freq=='percent':
                fid.write("  ['%s', '%s', '%.1f%%'], \n" % (mycommand, u[0], u[1]) )            
            else:
                fid.write("  ['%s', '%s', %d], \n" % (mycommand, u[0], u[1]) )
        fid.write('], \n')
    fid.write(']; \n')

def display_sys_gen_console_unmap(samples_pred):
    # Print model predictions after remapping
    #  There must also be a remapping from current tokens back to canonical tokens/
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predictions for each episode
    for idx,sample in enumerate(samples_pred):
        assert('unmap_input' in sample['aux']), "there must be mapping back to canonical text form"
        ui = lambda x : list(map(sample['aux']['unmap_input'], x))
        uo = lambda x : list(map(sample['aux']['unmap_output'], x))
        if 'filename' in sample['aux']:
            print('Evaluation episode ' + str(idx) + '; filename:', sample['aux']['filename'])
        else:
            print('Evaluation episode ' + str(idx))
        in_support = sample['in_support']
        is_novel = np.logical_not(in_support)
        if 'grammar' in sample:
            print("")
            print(sample['grammar'])
        print('  support items;')
        display_input_output(ui(sample['xs']),uo(sample['ys']),uo(sample['ys']))
        print('  retrieval items;',round(sample['acc_retrieve'],3),'% correct')
        display_input_output(extract(in_support,ui(sample['xq'])), extract(in_support,uo(sample['yq_predict'])), extract(in_support,uo(sample['yq'])))
        print('  generalization items;',round(sample['acc_novel'],3),'% correct')
        display_input_output(extract(is_novel,ui(sample['xq'])),extract(is_novel,uo(sample['yq_predict'])),extract(is_novel,uo(sample['yq'])))

def mean_masked_dim1(x,mask):
    # Compute the mean of non-masked tokens in the seq_len dimension (dim = 1)
    # Input
    # x : [batch_size,seq_len,d_model] tensor
    # mask : [batch_size,seq_len], such that True means a token is missing
    # Output
    # y : [batch_size,d_model] (mean of non-masked tokens, over seq_len dimension)
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be a boolean tensor")
    mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x) # [batch_size,seq_len,d_model]
    x = x.masked_fill(mask,0.0)
    y = x.sum(dim=1) # [batch_size,d_model]
    mask_sum = torch.logical_not(mask).sum(dim=1) # [batch_size,d_model]
    mask_sum = mask_sum.masked_fill(mask_sum == 0, 1.0)
    y = y / mask_sum
    return y

if __name__ == '__main__':
    # Test mean_masked
    x = torch.ones([2,3,4],dtype=torch.float32)
    print('x.shape',x.shape)
    x[0,0,:] = float('inf')
    x[1,1,:] = float('inf')
    x[1,2,:] = float('inf')
    
    mask = torch.zeros([2,3],dtype=torch.bool)
    mask[0,0] = True
    mask[1,1] = True
    mask[1,2] = True
    print('mask.shape',mask.shape)
    print('mask',mask)
    y = mean_masked_dim1(x,mask)
    print('y',y)
    assert(torch.all(y==torch.ones(y.shape))), 'Error in mean_masked_dim1'
    print('Test of mean_masked_dim1 passed')