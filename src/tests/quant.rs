use crate::config::BaseConfig;
use crate::models::q_llama::Which;
use crate::models::HubInfo;
use anyhow::{Error, Result};
use candle_examples::token_output_stream::TokenOutputStream;

#[tokio::test]
async fn test_tokenizer() -> Result<()> {
    let config = BaseConfig::<Which>::default();
    println!("{:?}", config);

    let tokenizer = config.setup_tokenizer().await?;
    println!("{:#?}", tokenizer.get_added_vocabulary());

    let tokens = [128011, 57668, 53901, 128012, 128013, 198];
    println!("{}", tokenizer.decode(&tokens, false).map_err(Error::msg)?);
    // 单个token-/>字符
    tokens.iter().for_each(|t| {
        print!("{}", tokenizer.decode(&[*t], true).unwrap());
    });
    println!();

    let mut tos = TokenOutputStream::new(tokenizer);
    tokens.iter().for_each(|t| {
        if let Some(t) = tos.next_token(*t).unwrap() {
            print!("{}", t);
        }
    });
    if let Some(t) = tos.decode_rest()? {
        print!("{}", t);
    }
    println!();

    Ok(())
}
