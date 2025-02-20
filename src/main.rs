use std::path::PathBuf;
use std::io::{self, Write};
use tokenizers::Tokenizer;

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;

fn main() {
    // 获取用户输入来选择模型
    let model_type = get_user_choice();

    if model_type == "story" {
        story_model();
    } else {
        chat_model();
    }
}

fn get_user_choice() -> String {
    println!("Please choose a model (story/chat):");

    let mut choice = String::new();
    io::stdout().flush().unwrap();  // Make sure the prompt is printed before user input
    io::stdin().read_line(&mut choice).unwrap();

    let choice = choice.trim().to_lowercase();

    // 检查输入是否有效
    if choice == "story" || choice == "chat" {
        return choice;
    } else {
        println!("Invalid choice. Defaulting to 'story' model.");
        return "story".to_string();  // Default to 'story' if input is invalid
    }
}


// ======== Story Model ========
fn story_model() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 在这里可以定义有关故事模型的特定操作
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("{}", input);

    // 调用生成函数
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

// ======== Chat Model ========
fn chat_model() {
    let chat_bot = chat::ChatModel::new();
    chat_bot.begin();
    // 在这里可以定义有关聊天模型的特定操作
}