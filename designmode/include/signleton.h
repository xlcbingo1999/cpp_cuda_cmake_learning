#include <string>
#include <mutex>

class Signle {
public:
    Signle(Signle& other) = delete;
    Signle& operator=(Signle& other) = delete;
    static Signle* getInstance(const std::string& value);
    std::string getValue() const;
private:
    static Signle* signleton_;
    static std::mutex signleton_mutex_;
    std::mutex val_mutex_;
    std::string _val = "";
    Signle(const std::string value);
};

