#include <rclcpp/rclcpp.hpp>
#include <audio_common_msgs/msg/audio_data.hpp>
#include <sndfile.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <fstream>

class AudioRecorderNode : public rclcpp::Node
{
public:
    AudioRecorderNode() : Node("audio_recorder_node")
    {
        RCLCPP_INFO(this->get_logger(), "Audio Recorder Node with Whisper transcription started.");
        sub_ = this->create_subscription<audio_common_msgs::msg::AudioData>(
            "/speech_audio", 10,
            std::bind(&AudioRecorderNode::audioCallback, this, std::placeholders::_1));
    }

private:
    void audioCallback(const audio_common_msgs::msg::AudioData::SharedPtr msg)
    {
        if (msg->data.empty()) return;

        // Create timestamped filename
        auto t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << "/home/robot/recordings/audio_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string base_filename = oss.str();
        std::string wav_filename = base_filename + ".wav";
        std::string txt_filename = base_filename + ".txt";

        // Save audio to WAV file
        if (!saveWav(msg->data, wav_filename)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save audio file.");
            return;
        }

        // Run Whisper.cpp transcription
        std::string transcript = runWhisper(wav_filename);
        if (transcript.empty()) {
            RCLCPP_WARN(this->get_logger(), "No transcription result, something went wrong.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Transcription is : %s", transcript.c_str());

        // Save transcription
        std::ofstream out(txt_filename);
        out << transcript; 
        out.close();

        RCLCPP_INFO(this->get_logger(), "Saved transcription to %s", txt_filename.c_str());
    }

    bool saveWav(const std::vector<uint8_t> &audio, const std::string &filename)
    {
        SF_INFO sfinfo;
        sfinfo.channels = 1;
        sfinfo.samplerate = 16000;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

        SNDFILE *outfile = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
        if (!outfile) return false;

         // Convert uint8_t to int16_t assuming 16-bit samples
        std::vector<short> buffer(audio.size() / 2);
        for (size_t i = 0; i < buffer.size(); ++i)
            buffer[i] = static_cast<short>(audio[2 * i] | (audio[2 * i + 1] << 8));

        sf_write_short(outfile, buffer.data(), buffer.size());
        sf_close(outfile);
        return true;
    }

    std::string runWhisper(const std::string &wav_path)
    {
        const std::string whisper_exec = "/home/robot/ament_ws/src/whisper.cpp/build/bin/whisper-cli";  
        const std::string model_path = "/home/robot/ament_ws/src/whisper.cpp/models/ggml-base.en.bin";  //can change to other model to tiny,small,medium etc

        //command to run whisper as cli
        std::ostringstream cmd;
        cmd << whisper_exec
            << " -m " << model_path
            << " -f " << wav_path
            << " --language en"
            << " --no-timestamps"
            << " -otxt"
            << " 2>/dev/null"; //to suppress all output (stderr and stdout)

        FILE *pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) {
            RCLCPP_ERROR(this->get_logger(), "Failed to execute whisper.cpp");
            return "";
        }

        char buffer[512];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            output += buffer;

        pclose(pipe);
        return output;
    }

    rclcpp::Subscription<audio_common_msgs::msg::AudioData>::SharedPtr sub_;
};
    
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AudioRecorderNode>());
    rclcpp::shutdown();
    return 0;
}
