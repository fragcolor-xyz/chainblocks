/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

use hmac::Hmac;
use pbkdf2::pbkdf2;
use sha2::Sha512;
use shards::core::register_shard;
use shards::shard::Shard;
use shards::types::common_type;
use shards::types::ClonedVar;
use shards::types::Context;
use shards::types::ParamVar;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::BYTES_TYPES;
use shards::types::STRING_TYPES;
use std::convert::TryInto;

#[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
use {
    jsonwebtoken::{decode, Algorithm, DecodingKey, Validation},
    jsonwebtoken::jwk::Jwk,
    jsonwebtoken::jwk::KeyAlgorithm,
    serde::{Deserialize, Serialize},
};

#[macro_use]
extern crate shards;

#[macro_use]
extern crate lazy_static;

pub mod argon;
pub mod chachapoly;
pub mod ecdsa;
pub mod hash;
pub mod signatures;

static CRYPTO_KEY_TYPES: &[Type] = &[common_type::bytes, common_type::bytes_var];

use bip39::{Language, Mnemonic, MnemonicType};

lazy_static! {
  pub static ref MNEMONIC_INPUT_TYPES: Vec<Type> = vec![common_type::int];
  pub static ref MNEMONIC_OUTPUT_TYPES: Vec<Type> = vec![common_type::string];
}

#[derive(shards::shard)]
#[shard_info("Mnemonic.Generate", "Generates a BIP39 mnemonic")]
struct MnemonicGenerate {
  output: ClonedVar,
}

impl Default for MnemonicGenerate {
  fn default() -> Self {
    Self {
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MnemonicGenerate {
  fn input_types(&mut self) -> &Types {
    &MNEMONIC_INPUT_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &MNEMONIC_OUTPUT_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let size: i64 = input.try_into().unwrap();
    let mnemonic_type = match size {
      12 => MnemonicType::Words12,
      15 => MnemonicType::Words15,
      18 => MnemonicType::Words18,
      21 => MnemonicType::Words21,
      24 => MnemonicType::Words24,
      _ => return Err("Invalid mnemonic size"),
    };
    let mnemonic = Mnemonic::new(mnemonic_type, Language::English);
    self.output = Var::ephemeral_string(mnemonic.phrase()).into();
    Ok(Some(self.output.0))
  }
}

#[derive(shards::shard)]
#[shard_info("Mnemonic.ToSeed", "Converts a BIP39 mnemonic to a seed")]
struct MnemonicToSeed {
  output: ClonedVar,
}

impl Default for MnemonicToSeed {
  fn default() -> Self {
    Self {
      output: ClonedVar::default(),
    }
  }
}

impl MnemonicToSeed {
  // for legacy reasons, we use the way substrate bip39 does it
  pub fn seed_from_entropy(entropy: &[u8], password: &str) -> Result<[u8; 64], &'static str> {
    if entropy.len() < 16 || entropy.len() > 32 || entropy.len() % 4 != 0 {
      return Err("Invalid entropy length");
    }

    let mut salt = String::with_capacity(8 + password.len());
    salt.push_str("mnemonic");
    salt.push_str(password);

    let mut seed = [0u8; 64];

    pbkdf2::<Hmac<Sha512>>(entropy, salt.as_bytes(), 2048, &mut seed)
      .map_err(|_| "PBKDF2 error")?;

    Ok(seed)
  }
}

#[shards::shard_impl]
impl Shard for MnemonicToSeed {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &BYTES_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let string: &str = input.try_into().unwrap();
    let mnemonic =
      Mnemonic::from_phrase(string, Language::English).map_err(|_| "Invalid mnemonic")?;
    let seed = Self::seed_from_entropy(mnemonic.entropy(), "").map_err(|_| "Invalid entropy")?;
    // let seed = Seed::new(&mnemonic, "");
    self.output = Var::ephemeral_slice(seed.as_slice()).into();
    Ok(Some(self.output.0))
  }
}

#[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
#[derive(Debug, Serialize, Deserialize)]
struct Claims {
  aud: String, // Optional. Audience
  exp: usize, // Required (validate_exp defaults to true in validation). Expiration time (as UTC timestamp)
  sub: String, // Optional. Subject (whom token refers to)
}

#[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
#[derive(shards::shard)]
#[shard_info("Jwt.Decode", "Decodes a JWT token")]
struct JwtDecode {
  output: ClonedVar,

  #[shard_param("Jwk", "The Key in JWK format to use for decoding the token.", [common_type::string, common_type::string_var])]
  jwk: ParamVar,

  #[shard_param("Audience", "The audience to use for decoding the token.", [common_type::string, common_type::string_var])]
  audience: ParamVar,
}

#[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
impl Default for JwtDecode {
  fn default() -> Self {
    Self {
      output: ClonedVar::default(),
      jwk: ParamVar::default(),
      audience: ParamVar::default(),
    }
  }
}

#[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
#[shards::shard_impl]
impl Shard for JwtDecode {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let jwk: &str = self.jwk.get().try_into().unwrap();
    let jwk = serde_json::from_str::<Jwk>(jwk).unwrap();
    let decoding_key = DecodingKey::from_jwk(&jwk).map_err(|e| {
      shlog_error!("Invalid JWK: {}", e);
      "Invalid JWK"
    })?;

    // Set up validation
    let algo = jwk.common.key_algorithm.ok_or("Unsupported key type")?;
    let mut validation = Validation::new(match algo {
      KeyAlgorithm::ES256 => Algorithm::ES256,
      KeyAlgorithm::ES384 => Algorithm::ES384,
      KeyAlgorithm::RS256 => Algorithm::RS256,
      KeyAlgorithm::RS384 => Algorithm::RS384,
      _ => return Err("Unsupported key type"),
    });
    let audience: &str = self.audience.get().try_into().unwrap();
    validation.set_audience(&[audience]);

    // Decode and verify the token
    let token: &str = input.try_into().unwrap();
    let token_data = decode::<Claims>(token, &decoding_key, &validation).map_err(|e| {
      shlog_error!("Invalid token: {}", e);
      "Invalid token"
    })?;

    self.output = Var::ephemeral_string(token_data.claims.sub.as_str()).into();
    Ok(Some(self.output.0))
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_crypto_crypto(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }

  ecdsa::register_shards();
  hash::register_shards();
  signatures::register_shards();
  chachapoly::register_shards();
  register_shard::<MnemonicGenerate>();
  register_shard::<MnemonicToSeed>();
  argon::register_shards();
  #[cfg(not(any(target_arch = "wasm32", target_os = "windows")))]
  register_shard::<JwtDecode>();
}
