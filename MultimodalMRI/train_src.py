from trainer import MRITrainer


if __name__ == '__main__':
    model = MRITrainer(
        model_name='src',
        device='cuda:0',
        batch_size=16,
        init_lr=0.001,
        zoom_dataset=True
    )
    model.train()
