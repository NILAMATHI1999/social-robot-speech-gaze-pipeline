#include <rclcpp/rclcpp.hpp>
#include <audio_common_msgs/msg/audio_data_stamped.hpp>
#include <audio_common_msgs/msg/audio_data.hpp>
#include <std_msgs/msg/string.hpp>
#include <whisper.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

class WhisperTranscriptionNode : public rclcpp::Node
{
public:
    WhisperTranscriptionNode() : Node("whisper_transcription_node")
    {
        // Declare parameters
        this->declare_parameter<std::string>("model_path", "models/ggml-base.en.bin");
        this->declare_parameter<int>("n_threads", 4);
        this->declare_parameter<std::string>("language", "en");
        this->declare_parameter<bool>("translate", false);
        this->declare_parameter<bool>("save_files", true);
        this->declare_parameter<std::string>("save_directory", "../../../recordings");
        this->declare_parameter<int>("sample_rate", 16000);
        
        // Get parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        n_threads_ = this->get_parameter("n_threads").as_int();
        language_ = this->get_parameter("language").as_string();
        translate_ = this->get_parameter("translate").as_bool();
        save_files_ = this->get_parameter("save_files").as_bool();
        save_directory_ = this->get_parameter("save_directory").as_string();
        sample_rate_ = this->get_parameter("sample_rate").as_int();
        
        // Create save directory if it doesn't exist
        if (save_files_) {
            mkdir(save_directory_.c_str(), 0755);
            RCLCPP_INFO(this->get_logger(), "Saving recordings to: %s", save_directory_.c_str());
        }
        
        RCLCPP_INFO(this->get_logger(), "Loading Whisper model from: %s", model_path.c_str());
        
        // Use new whisper_init_from_file_with_params API
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = true;  // Set to true if there is GPU support
        
        ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        
        if (ctx_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load Whisper model!");
            rclcpp::shutdown();
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Whisper model loaded successfully");
        
        // Create subscriber
        audio_sub_ = this->create_subscription<audio_common_msgs::msg::AudioData>(
            "/speech_audio", 10,
            std::bind(&WhisperTranscriptionNode::audioCallback, this, std::placeholders::_1));
    }
    
    ~WhisperTranscriptionNode()
    {
        if (ctx_ != nullptr) {
            whisper_free(ctx_);
        }
    }

private:
    std::string getTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
        ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    void saveWavFile(const std::string& filename, const std::vector<float>& pcm_data)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", filename.c_str());
            return;
        }
        
        // WAV header
        uint32_t data_size = pcm_data.size() * 2;  // 16-bit samples
        uint32_t file_size = 36 + data_size;
        uint16_t num_channels = 1;
        uint32_t sample_rate = sample_rate_;
        uint16_t bits_per_sample = 16;
        uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        uint16_t block_align = num_channels * bits_per_sample / 8;
        
        // RIFF header
        file.write("RIFF", 4);
        file.write(reinterpret_cast<const char*>(&file_size), 4);
        file.write("WAVE", 4);
        
        // fmt chunk
        file.write("fmt ", 4);
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1;  // PCM
        file.write(reinterpret_cast<const char*>(&fmt_size), 4);
        file.write(reinterpret_cast<const char*>(&audio_format), 2);
        file.write(reinterpret_cast<const char*>(&num_channels), 2);
        file.write(reinterpret_cast<const char*>(&sample_rate), 4);
        file.write(reinterpret_cast<const char*>(&byte_rate), 4);
        file.write(reinterpret_cast<const char*>(&block_align), 2);
        file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
        
        // data chunk
        file.write("data", 4);
        file.write(reinterpret_cast<const char*>(&data_size), 4);
        
        // Convert float32 back to int16 and write
        for (float sample : pcm_data) {
            int16_t int_sample = static_cast<int16_t>(sample * 32767.0f);
            file.write(reinterpret_cast<const char*>(&int_sample), 2);
        }
        
        file.close();
        //RCLCPP_INFO(this->get_logger(), "Saved audio to: %s", filename.c_str());
    }
    
    void saveTranscription(const std::string& filename, const std::string& text)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", filename.c_str());
            return;
        }
        
        file << text << std::endl;
        file.close();
        RCLCPP_INFO(this->get_logger(), "Saved transcription to: %s", filename.c_str());
    }
    
    void audioCallback(const audio_common_msgs::msg::AudioData::SharedPtr msg)
    {
         
        transcription_pub_ = this->create_publisher<std_msgs::msg::String>("transcription", 10);
    
        RCLCPP_INFO(this->get_logger(), "Received audio data of size: %zu bytes", msg->data.size());
        
        // Convert uint8_t audio data to float32 PCM format
        // Assuming 16-bit PCM input from ReSpeaker
        std::vector<float> pcm_data;
        size_t sample_count = msg->data.size() / 2;
        pcm_data.resize(sample_count);
        
        const int16_t* samples = reinterpret_cast<const int16_t*>(msg->data.data());
        for (size_t i = 0; i < sample_count; i++) {
            pcm_data[i] = static_cast<float>(samples[i]) / 32768.0f;
        }
        
        //RCLCPP_INFO(this->get_logger(), "Processing %zu samples with Whisper...", sample_count);
        
        // Run Whisper inference
        std::string transcription = transcribe(pcm_data);
        
        if (!transcription.empty()) {
            // Publish transcription
            auto transcription_msg = std_msgs::msg::String();
            transcription_msg.data = transcription;
            transcription_pub_->publish(transcription_msg);
            
            RCLCPP_INFO(this->get_logger(), "========================================");
            RCLCPP_INFO(this->get_logger(), "TRANSCRIPTION: %s", transcription.c_str());
            RCLCPP_INFO(this->get_logger(), "========================================");
            
            // Save files if enabled
            if (save_files_) {
                std::string timestamp = getTimestamp();
                std::string wav_filename = save_directory_ + "/audio_" + timestamp + ".wav";
                std::string txt_filename = save_directory_ + "/transcription_" + timestamp + ".txt";
                
                saveWavFile(wav_filename, pcm_data);
                saveTranscription(txt_filename, transcription);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No transcription result (empty or silence detected)");
        }
    }
    
    std::string transcribe(const std::vector<float>& pcm_data)
    {
        // Set up whisper parameters
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        
        wparams.n_threads = n_threads_;
        wparams.print_progress = false;
        wparams.print_special = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = false;
        wparams.translate = translate_;
        wparams.language = language_.c_str();
        wparams.no_context = true;
        wparams.single_segment = false;
        
        // Run inference
        if (whisper_full(ctx_, wparams, pcm_data.data(), pcm_data.size()) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to process audio");
            return "";
        }
        
        // Get transcription results
        std::string result;
        const int n_segments = whisper_full_n_segments(ctx_);
        
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx_, i);
            result += text;
        }
        
        // Trim whitespace
        size_t start = result.find_first_not_of(" \t\n\r");
        size_t end = result.find_last_not_of(" \t\n\r");
        
        if (start != std::string::npos && end != std::string::npos) {
            return result.substr(start, end - start + 1);
        }
        
        return result;
    }
    
    rclcpp::Subscription<audio_common_msgs::msg::AudioData>::SharedPtr audio_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr transcription_pub_;
    
    whisper_context* ctx_;
    int n_threads_;
    std::string language_;
    bool translate_;
    bool save_files_;
    std::string save_directory_;
    int sample_rate_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhisperTranscriptionNode>());
    rclcpp::shutdown();
    return 0;
}
