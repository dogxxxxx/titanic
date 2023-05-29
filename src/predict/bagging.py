def bagging(*args):
    add_lst = []
    for i in range(len(args[0])):
        sum_at_index = sum(arg[i] for arg in args)
        add_lst.append(sum_at_index)
    
    threshold = len(args) / 2
    result = [1 if val > threshold else 0 for val in add_lst]
    return result
