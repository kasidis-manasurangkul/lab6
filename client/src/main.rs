use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
use std::io::Write;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;
const MAX_CHUNK_SIZE: usize = 1024 * 1024 * 40;
const ACK_INTERVAL: usize = 1024 * 1024 * 100; // 100 MB

trait Model: Sized {
    fn new() -> Self;
    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn parameters(&self) -> Vec<&[f32]>;
}

#[derive(Debug, Clone)]
struct LinearModel {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Serialize for LinearModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LinearModel", 2)?;
        state.serialize_field("weights", &self.weights)?;
        state.serialize_field("biases", &self.biases)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for LinearModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = &["weights", "biases"];
        deserializer.deserialize_struct("LinearModel", fields, LinearModelVisitor)
    }
}

struct LinearModelVisitor;

impl<'de> serde::de::Visitor<'de> for LinearModelVisitor {
    type Value = LinearModel;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a LinearModel struct")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let weights: Vec<Vec<f32>> = seq
            .next_element()?
            .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        let biases: Vec<f32> = seq
            .next_element()?
            .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
        Ok(LinearModel { weights, biases })
    }
}

impl Model for LinearModel {
    fn new() -> Self {
        Self {
            weights: vec![vec![0.0; IMAGE_DIM]; LABELS],
            biases: vec![0.0; LABELS],
        }
    }

    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter()
            .map(|x| {
                self.weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(w, b)| x.iter().zip(w).map(|(xi, wi)| xi * wi).sum::<f32>() + b)
                    .collect()
            })
            .collect()
    }

    fn parameters(&self) -> Vec<&[f32]> {
        let mut params = Vec::new();
        params.extend(self.weights.iter().map(|w| w.as_slice()));
        params.push(self.biases.as_slice());
        params
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

fn model_train(
    mut model: LinearModel,
    train_images: &[Vec<f32>],
    train_labels: &[i64],
    learning_rate: f64,
    epochs: usize,
) -> LinearModel {
    for epoch in 1..=epochs {
        let logits = model.forward(train_images);
        let predictions: Vec<i64> = logits
            .iter()
            .map(|logit| {
                logit
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0 as i64
            })
            .collect();
        let accuracy = predictions
            .iter()
            .zip(train_labels)
            .filter(|(p, l)| p == l)
            .count() as f32
            / train_labels.len() as f32;

        println!(
            "Epoch {}/{} - Train Accuracy: {:.4}",
            epoch, epochs, accuracy,
        );

        // Update weights and biases (dummy update for simplicity)
        model.weights = model
            .weights
            .iter()
            .map(|w| w.iter().map(|wi| wi + learning_rate as f32).collect())
            .collect();
        model.biases = model
            .biases
            .iter()
            .map(|b| b + learning_rate as f32)
            .collect();
    }

    model
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingData {
    images: Vec<Vec<f32>>,
    labels: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableTrainingData(TrainingData);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to the server
    let mut client = TcpStream::connect("localhost:6979").await?;

    // Register with the server
    let register_message: Value = json!({
        "command": "Register",
        "model": "LinearModel",
    });
    client
        .write_all(register_message.to_string().as_bytes())
        .await?;
    client.write_all(b"<EOD>").await?;
    println!("Registered with the server.");

    let mut current_batch = 1;

    loop {
        println!("Waiting for the server to send a message...");
        // Receive message from the server
        let mut message_buffer = Vec::new();
        let mut end_of_data = false;

        while !end_of_data {
            let mut chunk = vec![0; MAX_CHUNK_SIZE];
            let n = client.read(&mut chunk).await?;

            if n == 0 {
                println!("Server disconnected.");
                return Ok(());
            }

            message_buffer.extend_from_slice(&chunk[..n]);

            if chunk[..n].ends_with(b"<EOD>") {
                end_of_data = true;
                message_buffer.truncate(message_buffer.len() - 5); // Remove the end-of-data marker from the buffer
            }
        }

        let message_data = std::str::from_utf8(&message_buffer)?;
        // println!("Received message: {}", message_data);
        println!("received message: {:?} bytes", message_data.len());

        if let Ok(json) = serde_json::from_str::<Value>(message_data) {
            if let Some(message) = json["message"].as_str() {
                match message {
                    "Waiting for more clients to join..." => {
                        // Wait for the server to send training data
                        println!("Waiting for more clients to join...");
                        continue;
                    }
                     "Starting new batch..." => {
                // Get the current batch number from the server's message
                if let Some(batch) = json["batch"].as_u64() {
                    current_batch = batch as usize;
                }
                println!("Starting new batch {}...", current_batch);

                // Deserialize the training data from the message
                if let Some(training_data) = json["training_data"].as_object() {
                    let data: Result<SerializableTrainingData, _> =
                        serde_json::from_value(serde_json::Value::Object(training_data.clone()));
                    match data {
                        Ok(data) => {
                            let train_images = &data.0.images;
                            let train_labels = &data.0.labels;

                            // Train the local model with the updated data
                            let training_args = TrainingArgs {
                                learning_rate: 0.1,
                                epochs: 1,
                            };
                            println!("Training with {} images...", train_images.len());
                            let model = LinearModel::new();
                            let trained_model = model_train(
                                model,
                                train_images,
                                train_labels,
                                training_args.learning_rate,
                                training_args.epochs,
                            );

                            // Send the trained model parameters (weights and biases) to the server
                            let model_message = json!({
                                "command": "UpdateModel",
                                "model": {
                                    "weights": trained_model.weights,
                                    "biases": trained_model.biases,
                                },
                                "batch": current_batch,
                            });
                            let model_message_str = serde_json::to_string(&model_message)?;
                            println!("Sending model parameters for batch {}:", current_batch);
                            client.write_all(model_message_str.as_bytes()).await?;

                            // Send the end-of-data marker after sending the model parameters
                            client.write_all(b"<EOD>").await?;

                            println!("Model parameters sent successfully.");
                        }
                        Err(e) => {
                            println!("Failed to deserialize the received data: {}", e);
                            continue;
                        }
                    }
                } else {
                    println!("Training data not found in the message.");
                    continue;
                }
            }
                    "All batches completed. Disconnecting..." => {
                        println!("All batches completed. Disconnecting...");
                        client.shutdown().await?;
                        return Ok(());
                    }
                    _ => {
                        // Handle other messages if needed
                        println!("Received message: {}", message);
                    }
                }
            }
        }
    }

    // Close the connection
    client.shutdown().await?;

    Ok(())
}
