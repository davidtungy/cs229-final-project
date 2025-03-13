from utils import schwab_utils, tradeui_utils
from analysis import technical_analysis, transformer

def main():
    train_x, train_y, val_x, val_y = technical_analysis.run()
    transformer.run(train_x, train_y, val_x, val_y)


if __name__ == "__main__":
    main()