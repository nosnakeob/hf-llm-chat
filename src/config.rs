use crate::models::{FromGGUF, HubInfo};
use crate::utils::load::{load_gguf, load_tokenizer};
use anyhow::Result;
use candle::Device;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct BaseConfig<W> {
    /// The length of the sample to generate (in tokens).
    pub sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,

    /// The seed to use when generating random samples.
    pub seed: u64,

    /// The device to use for inference.
    pub device: Device,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,

    /// The model to use.
    pub which: W,
}

impl<W: Default> Default for BaseConfig<W> {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            // !有性能影响
            top_p: None,
            seed: 299792458,
            device: candle_examples::device(false).unwrap(),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            which: W::default(),
        }
    }
}

impl<Wi: HubInfo> BaseConfig<Wi> {
    pub async fn setup_model(&self) -> Result<(Wi::ModelWeight, u32)> {
        let info = self.which.info();
        let (mut file, ct) = load_gguf(info.model_repo, info.model_file).await?;
        let eos_token_id = ct.metadata.get("tokenizer.ggml.eos_token_id").unwrap().to_u32()?;
        let model = Wi::ModelWeight::from_gguf(ct, &mut file, &self.device)?;

        Ok((model, eos_token_id))
    }

    pub async fn setup_tokenizer(&self) -> Result<Tokenizer> {
        load_tokenizer(self.which.info().tokenizer_repo).await
    }
}
