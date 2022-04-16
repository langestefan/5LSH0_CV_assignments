from wrappers import train, eval

if __name__ == '__main__':
    # Define your own paths here
    configs = [
        # './configs/broad_config.yaml',
        # './configs/std_config.yaml',
        # './configs/deep_config.yaml',
        './configs/custom_config.yaml'
    ]
    # For each config file we will train and generate songs
    for c in configs:
        print('Training: {}'.format(c))
        train.train(c)
        print('Testing: {}'.format(c))
        eval.test(c)

