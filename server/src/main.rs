use candle_core::DType;
use candle_datasets::vision;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Mutex};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;
const BATCH_SIZE: usize = 1;
const BATCH_COUNT: usize = 3;
const MAX_CHUNK_SIZE: usize = 1024 * 1024 * 40;
const ACK_INTERVAL: usize = 1024 * 1024 * 100; // 100 MB

trait Model: Sized {
    fn new() -> Self;
    fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>>;
    fn parameters(&self) -> Vec<&[f32]>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearModel {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingData {
    images: Vec<Vec<f32>>,
    labels: Vec<i64>,
}

struct ServerState {
    global_model: Option<LinearModel>,
    client_models: Vec<Option<LinearModel>>,
    train_data: Arc<TrainingData>,
    current_batch: usize,
    connected_clients: usize,
    ready_clients: usize, // Add this field
    finished_clients: usize,
}

async fn process(
    mut socket: TcpStream,
    state: Arc<Mutex<ServerState>>,
    sender: broadcast::Sender<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client_address = socket.peer_addr()?.to_string();

    let mut receiver = sender.subscribe();

    {
        let mut server_state = state.lock().await;
        server_state.connected_clients += 1;
    }

    loop {
        let mut buffer = Vec::new();
        let mut end_of_data = false;

        while !end_of_data {
            let mut chunk = vec![0u8; MAX_CHUNK_SIZE];
            let n = socket.read(&mut chunk).await?;

            if n == 0 {
                println!("Client disconnected: {}", client_address);
                break;
            }

            buffer.extend_from_slice(&chunk[..n]);

            if chunk[..n].ends_with(b"<EOD>") {
                end_of_data = true;
                buffer.truncate(buffer.len() - 5);
            }
        }

        if !end_of_data {
            println!("Error: End-of-data marker not received.");
            continue;
        }

        if let Ok(json) = serde_json::from_slice::<Value>(&buffer) {
            if let Some(command) = json["command"].as_str() {
                match command {
                    "Register" => {
                        println!("Received registration request");
                        if let Some(model_name) = json["model"].as_str() {
                            if model_name == "LinearModel" {
                                let mut server_state = state.lock().await;
                                if server_state.connected_clients >= BATCH_SIZE {
                                    if server_state.current_batch <= BATCH_COUNT {
                                        println!("All clients have finished updating. Starting new batch...");

                                        // Create a single message containing the command and training data
                                        let global_model = server_state
                                            .global_model
                                            .clone()
                                            .unwrap_or_else(LinearModel::new);
                                        let model_weights = global_model.weights;
                                        let model_biases = global_model.biases;

                                        let message = json!({
                                            "message": "Starting new batch...",
                                            "batch": server_state.current_batch,
                                            "training_data": {
                                                "images": server_state.train_data.images,
                                                "labels": server_state.train_data.labels,
                                                "weights": model_weights,
                                                "biases": model_biases,
                                            }
                                        })
                                        .to_string();

                                        // Send the message to the client
                                        socket.write_all(message.as_bytes()).await?;
                                        socket.write_all(b"<EOD>").await?;

                                        // Reset the finished_clients count
                                        server_state.finished_clients = 0;
                                    } else {
                                        let response = json!({
                                            "message": "All batches completed. Disconnecting..."
                                        })
                                        .to_string();

                                        // Send the "All batches completed" command to the client
                                        socket.write_all(response.as_bytes()).await?;
                                        socket.write_all(b"<EOD>").await?;
                                    }
                                } else {
                                    let response =
                                        json!({ "message": "Waiting for more clients to join..." })
                                            .to_string();
                                    socket.write_all(response.as_bytes()).await?;
                                    socket.write_all(b"<EOD>").await?;
                                }
                            } else {
                                let response = json!({ "error": "Unsupported model" }).to_string();
                                socket.write_all(response.as_bytes()).await?;
                            }
                        } else {
                            let response = json!({ "error": "Missing model name" }).to_string();
                            socket.write_all(response.as_bytes()).await?;
                        }
                    }
                    "UpdateModel" => {
                        println!("Received model update");
                        if let Some(model_params) = json["model"].as_object() {
                            if let (Some(weights), Some(biases)) = (
                                model_params["weights"].as_array(),
                                model_params["biases"].as_array(),
                            ) {
                                let model_parameters = LinearModel {
                                    weights: weights
                                        .iter()
                                        .map(|w| {
                                            w.as_array()
                                                .unwrap()
                                                .iter()
                                                .map(|v| v.as_f64().unwrap() as f32)
                                                .collect()
                                        })
                                        .collect(),
                                    biases: biases
                                        .iter()
                                        .map(|b| b.as_f64().unwrap() as f32)
                                        .collect(),
                                };
                                println!("Updating model...");
                                let mut server_state = state.lock().await;

                                let client_index = server_state
                                    .client_models
                                    .iter()
                                    .position(|model| model.is_none())
                                    .unwrap_or(server_state.client_models.len());

                                if client_index < server_state.client_models.len() {
                                    server_state.client_models[client_index] =
                                        Some(model_parameters);
                                } else {
                                    server_state.client_models.push(Some(model_parameters));
                                }

                                println!("Connected clients: {}", server_state.connected_clients);
                                println!(
                                    "check {}",
                                    server_state
                                        .client_models
                                        .iter()
                                        .filter(|model| model.is_some())
                                        .count()
                                );

                                // Check if all clients have sent their model updates
                                if server_state
                                    .client_models
                                    .iter()
                                    .filter(|model| model.is_some())
                                    .count()
                                    == server_state.connected_clients
                                {
                                    // Average the model parameters from all clients
                                    let mut avg_weights = vec![vec![0.0; IMAGE_DIM]; LABELS];
                                    let mut avg_biases = vec![0.0; LABELS];
                                    for model in &server_state.client_models {
                                        if let Some(model) = model {
                                            for (avg_weight, model_weight) in
                                                avg_weights.iter_mut().zip(model.weights.iter())
                                            {
                                                for (avg_w, model_w) in
                                                    avg_weight.iter_mut().zip(model_weight.iter())
                                                {
                                                    *avg_w += model_w;
                                                }
                                            }
                                            for (avg_bias, model_bias) in
                                                avg_biases.iter_mut().zip(model.biases.iter())
                                            {
                                                *avg_bias += model_bias;
                                            }
                                        }
                                    }
                                    let num_clients = server_state.connected_clients as f32;
                                    for avg_weight in avg_weights.iter_mut() {
                                        for avg_w in avg_weight.iter_mut() {
                                            *avg_w /= num_clients;
                                        }
                                    }
                                    for avg_bias in avg_biases.iter_mut() {
                                        *avg_bias /= num_clients;
                                    }

                                    // Update the global model with the averaged parameters
                                    server_state.global_model = Some(LinearModel {
                                        weights: avg_weights,
                                        biases: avg_biases,
                                    });
                                    println!("Global model updated");

                                    // Clear the client models for the next batch
                                    server_state.client_models.clear();

                                    // Increment the finished_clients count
                                    server_state.finished_clients += 1;

                                    // Start a new batch
                                    server_state.current_batch += 1;
                                    if server_state.current_batch <= BATCH_COUNT {
                                        println!("All clients have finished updating. Starting new batch...");

                                        // Create a single message containing the command and training data
                                        let global_model = server_state
                                            .global_model
                                            .clone()
                                            .unwrap_or_else(LinearModel::new);
                                        let model_weights = global_model.weights;
                                        let model_biases = global_model.biases;

                                        let message = json!({
                                            "message": "Starting new batch...",
                                            "batch": server_state.current_batch,
                                            "training_data": {
                                                "images": server_state.train_data.images,
                                                "labels": server_state.train_data.labels,
                                                "weights": model_weights,
                                                "biases": model_biases,
                                            }
                                        })
                                        .to_string();

                                        // Send the message to the client
                                        socket.write_all(message.as_bytes()).await?;
                                        socket.write_all(b"<EOD>").await?;

                                        // Reset the finished_clients count
                                        server_state.finished_clients = 0;
                                    } else {
                                        let response = json!({
                                            "message": "All batches completed. Disconnecting..."
                                        })
                                        .to_string();

                                        // Send the "All batches completed" command to the client
                                        socket.write_all(response.as_bytes()).await?;
                                        socket.write_all(b"<EOD>").await?;
                                    }
                                }

                                // Unlock the server state
                                drop(server_state);
                            } else {
                                let response =
                                    json!({ "error": "Invalid model parameters" }).to_string();
                                socket.write_all(response.as_bytes()).await?;
                            }
                        } else {
                            let response =
                                json!({ "error": "Invalid model parameters" }).to_string();
                            socket.write_all(response.as_bytes()).await?;
                        }
                    }
                    _ => {
                        let response = json!({ "error": "Invalid command" }).to_string();
                        socket.write_all(response.as_bytes()).await?;
                    }
                }
            } else {
                let response = json!({ "error": "Missing command" }).to_string();
                socket.write_all(response.as_bytes()).await?;
            }
        }
    }

    {
        let mut server_state = state.lock().await;
        server_state.connected_clients -= 1;
    }

    Ok(())
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let listener = match TcpListener::bind("localhost:6979").await {
        Ok(listener) => {
            println!("Listening on: {}", listener.local_addr().unwrap());
            listener
        }
        Err(e) => {
            eprintln!("Failed to bind to port: {}", e);
            return Ok(());
        }
    };

    let mnist_data = vision::mnist::load()?;
    let train_data = Arc::new(TrainingData {
        images: mnist_data
            .train_images
            .to_vec2::<f32>()?
            .iter()
            .map(|row| row.to_vec())
            .collect(),
        labels: mnist_data.train_labels.to_dtype(DType::I64)?.to_vec1()?,
    });

    let server_state = Arc::new(Mutex::new(ServerState {
        global_model: Some(LinearModel::new()),
        client_models: vec![None; BATCH_SIZE],
        train_data,
        current_batch: 1,
        connected_clients: 0,
        ready_clients: 0, // Initialize ready_clients to 0
        finished_clients: 0,
    }));

    let (sender, _) = broadcast::channel(10);

    loop {
        let (socket, _) = match listener.accept().await {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
                continue;
            }
        };

        let state = Arc::clone(&server_state);
        let sender = sender.clone();
        println!("Accepted connection from: {}", socket.peer_addr()?);
        tokio::spawn(async move {
            if let Err(e) = process(socket, state, sender).await {
                eprintln!("Error processing connection: {}", e);
            }
        });
    }
}
