package com.example.demoSpringAI;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class DemoSpringAiApplication {

	public static void main(String[] args) {
		SpringApplication.run(DemoSpringAiApplication.class, args);
	}


//    @Bean
//    public CommandLineRunner commandLineRunner(ChatClient.Builder builder) {
//        return args -> {
//            var chatClient = builder.build();
//            String response = chatClient.prompt("What is Java?")
//                    .call()
//                    .content();
//
//            System.out.println(response);
//        };
//    }
}
