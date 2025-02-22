// src/chat.rs 基础模板
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::path::PathBuf;
use crate::model::Llama;
use crate::kvcache::KVCache;
struct Conversation {
    input: String,
    output: String,
}
struct Session{
    // 会话标题
    title: String,
    // 会话历史
    history: Vec<(usize, Conversation)>,
    // 会话缓存
    kvcache: KVCache<f32>,
}

pub struct ChatModel {
    // 模型参数
    model : Llama<f32>,
    tokenizer : Tokenizer,
    
    sessions : Vec<Session>,
    // 聊天模型特有字段
    system_prompt: String,

    // ...
}
/* 
    聊天模型的实现
        外部接口
            1. 实现ChatModel::new方法，用于初始化聊天模型
            2. generator trait实现
            3. 实现ChatModel::process_history方法，用于处理对话历史，会话回滚
            4. 实现会话切换，多会话管理
        内部接口
            1. ChatModel::format_prompt方法，用于格式化用户输入

*/

impl ChatModel {

    pub fn new() -> Self {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        
        // 先创建临时model实例
        let model = Llama::<f32>::from_safetensors(&model_dir);

        // 现在可以安全初始化结构体字段
        ChatModel {
            model,
            tokenizer: Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
            sessions: Vec::new(),
            system_prompt: "You are a helpful assistant".to_string(),
        }
    }
    pub fn begin(&mut self) {
        // let mut sessions: HashMap<String, Session> = HashMap::new();
        
        println!("Welcome to the chatbot!");
        loop {
            println!("Choose an option:");
            println!("1. Start a new conversation");
            println!("2. Continue a previous conversation");
            println!("3. Exit");
            print!("Enter your choice : ");
            io::stdout().flush().unwrap();
            
            let mut choice = String::new();
            io::stdin().read_line(&mut choice).unwrap();
            let choice = choice.trim();
    
            match choice {
                "1" => {
                    // 开启新会话
                    self.new_session();
                }
                "2" => {
                    // 继续之前的会话
                    self.history_session();
                }
                "3" => {
                    println!("Goodbye!");
                    break;
                }
                _ => {
                    println!("Invalid choice. Please choose again.");
                }
            }
        }
    }

    fn generator(&mut self, session_id: usize) {
        
        {
            let session = self.sessions.get_mut(session_id).unwrap();
            
            if session.history.is_empty() {
                println!("Hello, I am a chatbot");
            } else {
                println!("Welcome back to the conversation");
                for (_, conversation) in &session.history {
                    todo!("历史信息展示错误修复");
                    println!("User: {}", conversation.input);
                    println!("Assistant: {}", conversation.output);
                }
            }
        }
        loop {
            print!("User: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
    
            if input.trim() == "exit" {
                println!("Goodbye!");
                break;
            }
    
            let format_input = self.format_prompt(input.clone());
            let binding = self.tokenizer.encode(format_input.clone(), true).unwrap();
            let input_ids = binding.get_ids();

            let session = self.sessions.get_mut(session_id).unwrap();
            if session.history.is_empty() {
                session.title = format_input.clone();
            }

            // 调用生成函数
            let output_ids = self.model.chat_generator(
                input_ids,
                100,
                0.8,
                4,
                1.,
                &mut session.kvcache,
            );
            
            let output = self.tokenizer.decode(&output_ids, true).unwrap();
            let conversation = Conversation {
                input: input,
                output: output.clone(),
            };
            session.history.push((input_ids.len() + output_ids.len() , conversation));
            println!("Assistant: {}", output);
        }
    }

    fn new_session(&mut self) {
        // 调用生成函数
        // ...
        self.sessions.push(Session {
            title: "".to_string(),
            history: Vec::new(),
            kvcache: self.model.new_cache(),
        });
        

        let session_id = self.sessions.len() - 1;
        self.generator(session_id);
    }

    fn history_session(&mut self) {
        loop {
            println!("Choose a session to view:");
            for (i, session) in self.sessions.iter().enumerate() {
                println!("{}. Session {}", i + 1, session.title);
            }
            println!("Type 'exit' to return to the previous menu");
            print!("Enter your choice: ");
            io::stdout().flush().unwrap();
            
            let mut choice = String::new();
            io::stdin().read_line(&mut choice).unwrap();
            let choice = choice.trim();
            
            if choice.to_lowercase() == "exit" {
                println!("Returning to the main menu...");
                // 返回主菜单或其他逻辑
                return;
            }
            
            let choice: Result<usize, _> = choice.parse();
            match choice {
                Ok(choice) => {
                    if choice > 0 && choice <= self.sessions.len() {
                        let session_id = choice - 1;
                        self.generator(session_id);
                    } else {
                        println!("Invalid choice. Please select a valid session number or type 'exit'");
                    }
                }
                Err(_) => {
                    println!("Invalid input. Please enter a valid number or type 'exit'");
                }
            }
        }
    }

    fn format_prompt(&self, user_input: String) -> String {
        // 实现聊天专用的prompt格式
        let mut prompt = String::from("<|im_start|>User ");
        prompt.push_str(&user_input);
        prompt.push_str(&String::from("<|im_end|>\n<|im_start|>Assistant "));
        prompt
    }

}