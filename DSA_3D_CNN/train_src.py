from trainer import DSA3DCNNTrainer


if __name__ == '__main__':
    model = DSA3DCNNTrainer(
        model_name='src',
        device='cuda:0',
        batch_size=16,
        init_lr=0.000015
    )
    model.train(num_epoch=100)
