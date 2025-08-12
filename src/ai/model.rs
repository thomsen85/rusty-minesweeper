use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::MseLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    train::RegressionOutput,
};

use crate::constants;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    width: usize,
    height: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // Takes in two channels, one for opened and second for values.
            conv1: Conv2dConfig::new([2, 4], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([4, 4], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(4 * self.width * self.height, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.width * self.height).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Boards [batch_size, depth, height, width]
    ///   - Output [batch_size, height, width] Chance of mine
    pub fn forward(&self, boards: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, depth, height, width] = boards.dims();

        // TODO: Remove on release
        assert_eq!(constants::COLS, width);
        assert_eq!(constants::ROWS, height);
        assert_eq!(2, depth);

        // Create a channel at the second dimension.
        // let x = boards.reshape([batch_size, height, width, depth]);

        let x = self.conv1.forward(boards); // [batch_size, 4, width, height]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 4, width, height]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        // let x = self.pool.forward(x); // [batch_size, 16, 8, 8] - Im not pooling here.
        let x = x.reshape([batch_size, 4 * height * width]);

        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.linear2.forward(x); // [batch_size, num_classes]
        assert_eq!(x.dims(), [batch_size, height * width]);
        x
    }

    pub fn forward_regression(
        &self,
        boards: Tensor<B, 4>,
        mines: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
        let output = self.forward(boards);
        let mines_reshaped = mines.reshape(output.dims());
        let loss = MseLoss::new().forward(
            output.clone(),
            mines_reshaped.clone(),
            nn::loss::Reduction::Mean,
        );

        RegressionOutput::new(loss, output, mines_reshaped)
    }
}
