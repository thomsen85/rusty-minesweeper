use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::MseLoss,
        Dropout, DropoutConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    train::RegressionOutput,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    dropout: Dropout,
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
            conv3: Conv2dConfig::new([4, 1], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Boards [batch_size, depth, height, width]
    ///   - Output [batch_size, height, width] Chance of mine
    pub fn forward(&self, boards: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _depth, height, width] = boards.dims();

        let x = self.conv1.forward(boards); // [batch_size, 4, width, height]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 4, width, height]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x); // [batch_size, 1, width, height]
        x.reshape([batch_size, height * width])
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
