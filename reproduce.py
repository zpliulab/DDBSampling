from Arguments import parser
if __name__ == '__main__':
    args = parser.parse_args()
    if args.classifier == 'Seq-AE': # Corresponding to Seq-Deep in paper
        from train_AE import train_cv, val_cv, train_c1c2c3, val_c1c2c3

    else:
        from train import train_cv, val_cv, train_c1c2c3, val_c1c2c3

    if args.validation_strategy == 'CV':

        train_cv(args)
        val_cv(args, 'val')
        # val_cv(args, 'test_n')
    if args.validation_strategy == 'c1c2c3':
        train_c1c2c3(args)
        val_c1c2c3(args, 'val')
        val_c1c2c3(args, 'test')