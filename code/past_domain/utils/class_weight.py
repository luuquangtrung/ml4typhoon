def class_weight(num_pos: int, num_neg: int):
    total = num_pos + num_neg
    weight_neg = (total) / (2.0 * num_neg)
    weight_pos = (total) / (2.0 * num_pos)

    class_weights = [weight_neg, weight_pos]
    
    return class_weights

