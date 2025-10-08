package com.example.demoSpringAI.Controller;

import com.example.demoSpringAI.Model.Movie;
import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.document.Document;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.image.ImageResponse;
import org.springframework.ai.openai.*;
import org.springframework.ai.openai.api.OpenAiAudioApi;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.util.MimeTypeUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@RestController
@RequestMapping("/api")
public class GeminiController {

    @Autowired
    private VectorStore vectorStore;

    private ChatClient chatClient;

    @Autowired
    private OpenAiImageModel  openAiImageModel;

    @Autowired
    private OpenAiAudioTranscriptionModel  openAiAudioTranscriptionModel;

    @Autowired
    private OpenAiAudioSpeechModel  openAiAudioSpeechModel;

    ChatMemory chatMemory = MessageWindowChatMemory.builder().build();
    public GeminiController(ChatClient.Builder builder) {
        this.chatClient = builder
                .defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build())
                .build();
    }

    @GetMapping("/{question}")
    public String askGemini(@PathVariable String question) {
        String response = chatClient
                .prompt(question)
                .call()
                .content();

        return response;
    }

    @PostMapping("/recommend")
    public String recommendGemini(@RequestParam String type, @RequestParam String year, @RequestParam String lang) {
        String tempt = """
                    I want to watch a {type} movie. It should be around {year} 
                    year. And i want the movie to be in {lang} lang. Suggest only
                    one movie.
                    
                    Please give response in this format
                    1. Movie name
                    2. Cast
                    3. Basic Plot
                    4. Runtime
                    5. IMDB Rating
                """;

        PromptTemplate promptTemplate = new PromptTemplate(tempt);
        Prompt prompt = promptTemplate.create(Map.of("type", type, "year", year, "lang", lang));

        return chatClient.prompt(prompt).call().content();
    }

    @PostMapping("/product")
    public List<Document> recommendProduct(@RequestParam String word) {
      return vectorStore.similaritySearch(SearchRequest.builder().query(word).topK(2).build());
    }

    @PostMapping("/ask")
    public String askRAG(@RequestParam String question) {
        String response = chatClient
                .prompt(question)
                .advisors(new QuestionAnswerAdvisor(vectorStore))
                .call()
                .content();

        return response;
    }

    @GetMapping("/img/{question}")
    public String genImage(@PathVariable String question) {

        ImagePrompt imgPrompt = new ImagePrompt(question);

        ImageResponse response = openAiImageModel.call(imgPrompt);

        return response.getResult().getOutput().getUrl();
    }

    @PostMapping("/img/describe")
    public String describeImage(@RequestParam String text, @RequestParam MultipartFile image) {
        return chatClient.prompt()
                .user(us -> us.text(text)
                        .media(MimeTypeUtils.IMAGE_JPEG, image.getResource())
                )
                .call()
                .content();
    }

    @PostMapping("audio/stt")
    public String describeAudio(@RequestParam MultipartFile file) {

        OpenAiAudioTranscriptionOptions options = OpenAiAudioTranscriptionOptions
                .builder()
                .language("en")
                .responseFormat(OpenAiAudioApi.TranscriptResponseFormat.SRT)
                .build();

        AudioTranscriptionPrompt prompt = new AudioTranscriptionPrompt(file.getResource(), options);

        return openAiAudioTranscriptionModel.call(prompt).getResult().getOutput();
    }

    @PostMapping("audio/tts")
    public byte[] describeAudio(@RequestParam String text) {
        return openAiAudioSpeechModel.call(text);
    }


    @GetMapping("/movie")
    public Movie getMovie(@RequestParam String actor) {
        String message = """
                Give me the best movie of {actor} 
                in this {format}.
                """;

        BeanOutputConverter<Movie> opCon =  new BeanOutputConverter<>(Movie.class);

        PromptTemplate promptTemplate = new PromptTemplate(message);

        Prompt prompt = promptTemplate.create(Map.of("actor", actor, "format", opCon.getFormat()));
        return opCon.convert(Objects.requireNonNull(chatClient.prompt(prompt).call().content()));

    }

    @GetMapping("/movieList")
    public List<Movie> getMovieList(@RequestParam String actor) {
        String message = """
                Give me a list of movie of {actor} 
                in this {format}.
                """;

        BeanOutputConverter<List<Movie>> opCon =  new BeanOutputConverter<>(new ParameterizedTypeReference<List<Movie>>() {});

        PromptTemplate promptTemplate = new PromptTemplate(message);

        Prompt prompt = promptTemplate.create(Map.of("actor", actor, "format", opCon.getFormat()));
        List<Movie> movies = opCon.convert(Objects.requireNonNull(chatClient.prompt(prompt).call().content()));

        return movies;
    }
}
