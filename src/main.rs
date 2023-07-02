mod task_1;
mod task_2;
mod task_3;
mod task_4;

use std::io::{stdin, stdout, Write};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    loop {
        println!("Which task (1-4) to run? (exit typing 0): ");
        stdout().flush().unwrap();

        let mut buffer = String::new();
        stdin().read_line(&mut buffer).unwrap();

        match buffer.trim().parse::<usize>() {
            Ok(num) => match num {
                0 => break,
                1 => task_1::run(),
                2 => task_2::run(),
                3 => task_3::run(),
                4 => task_4::run(),
                _ => println!("Invalid number!"),
            }
            Err(_) => println!("Invalid input!"),
        }
    }

    Ok(())
}