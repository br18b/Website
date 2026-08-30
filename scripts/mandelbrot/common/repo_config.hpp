#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace mandelbrot::repo {

namespace fs = std::filesystem;

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    using Value = std::variant<std::nullptr_t, bool, long double, std::string, Array, Object>;

    Json() : value_(nullptr) {}
    explicit Json(Value value) : value_(std::move(value)) {}

    bool is_object() const { return std::holds_alternative<Object>(value_); }
    bool is_array() const { return std::holds_alternative<Array>(value_); }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_number() const { return std::holds_alternative<long double>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }

    const Object& object() const { return std::get<Object>(value_); }
    const Array& array() const { return std::get<Array>(value_); }
    const std::string& string() const { return std::get<std::string>(value_); }
    long double number() const { return std::get<long double>(value_); }
    bool boolean() const { return std::get<bool>(value_); }

    const Json* find(const std::string& key) const {
        if (!is_object()) return nullptr;
        const auto it = object().find(key);
        return it == object().end() ? nullptr : &it->second;
    }

    static Json parse(const std::string& text) {
        Parser parser(text);
        Json result = parser.value();
        parser.ws();
        if (!parser.done()) throw std::runtime_error("Unexpected trailing JSON content");
        return result;
    }

private:
    class Parser {
    public:
        explicit Parser(const std::string& text) : text_(text) {}
        bool done() const { return pos_ == text_.size(); }
        void ws() { while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
        Json value() {
            ws();
            if (pos_ >= text_.size()) fail("Unexpected end of JSON");
            switch (text_[pos_]) {
                case '{': return object_value();
                case '[': return array_value();
                case '"': return Json(string_value());
                case 't': literal("true"); return Json(true);
                case 'f': literal("false"); return Json(false);
                case 'n': literal("null"); return Json(nullptr);
                default: return Json(number_value());
            }
        }
    private:
        [[noreturn]] void fail(const std::string& message) const {
            throw std::runtime_error(message + " at JSON byte " + std::to_string(pos_));
        }
        void expect(char ch) {
            ws();
            if (pos_ >= text_.size() || text_[pos_] != ch) fail(std::string("Expected '") + ch + "'");
            ++pos_;
        }
        void literal(const char* value) {
            const std::string token(value);
            if (text_.compare(pos_, token.size(), token) != 0) fail("Invalid JSON literal");
            pos_ += token.size();
        }
        std::string string_value() {
            expect('"');
            std::string out;
            while (pos_ < text_.size()) {
                char ch = text_[pos_++];
                if (ch == '"') return out;
                if (ch != '\\') { out.push_back(ch); continue; }
                if (pos_ >= text_.size()) fail("Incomplete JSON escape");
                const char esc = text_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) fail("Incomplete unicode escape");
                        unsigned value = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char h = text_[pos_++];
                            value <<= 4;
                            if (h >= '0' && h <= '9') value += h - '0';
                            else if (h >= 'a' && h <= 'f') value += 10 + h - 'a';
                            else if (h >= 'A' && h <= 'F') value += 10 + h - 'A';
                            else fail("Invalid unicode escape");
                        }
                        if (value <= 0x7f) out.push_back(static_cast<char>(value));
                        else if (value <= 0x7ff) {
                            out.push_back(static_cast<char>(0xc0 | (value >> 6)));
                            out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
                        } else {
                            out.push_back(static_cast<char>(0xe0 | (value >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
                            out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
                        }
                        break;
                    }
                    default: fail("Invalid JSON escape");
                }
            }
            fail("Unterminated JSON string");
        }
        long double number_value() {
            ws();
            const std::size_t start = pos_;
            if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
            if (pos_ >= text_.size()) fail("Invalid JSON number");
            if (text_[pos_] == '0') ++pos_;
            else {
                if (!std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("Invalid JSON number");
                while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
            }
            if (pos_ < text_.size() && text_[pos_] == '.') {
                ++pos_;
                while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
            }
            if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
                ++pos_;
                if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
                while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
            }
            try { return std::stold(text_.substr(start, pos_ - start)); }
            catch (...) { fail("Invalid JSON number"); }
        }
        Json object_value() {
            expect('{');
            Object out;
            ws();
            if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; return Json(std::move(out)); }
            while (true) {
                ws();
                if (pos_ >= text_.size() || text_[pos_] != '"') fail("Expected object key");
                std::string key = string_value();
                expect(':');
                out.emplace(std::move(key), value());
                ws();
                if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; break; }
                expect(',');
            }
            return Json(std::move(out));
        }
        Json array_value() {
            expect('[');
            Array out;
            ws();
            if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; return Json(std::move(out)); }
            while (true) {
                out.push_back(value());
                ws();
                if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; break; }
                expect(',');
            }
            return Json(std::move(out));
        }
        const std::string& text_;
        std::size_t pos_ = 0;
    };
    Value value_;
};

inline fs::path canonical_or_absolute(fs::path path) {
    try { return fs::weakly_canonical(path); }
    catch (...) { return fs::absolute(path); }
}

inline fs::path resolve_data_root(
    const fs::path& explicit_data_root,
    const fs::path& project_root
) {
    fs::path selected = explicit_data_root;
    if (selected.empty()) {
        if (const char* environment = std::getenv("MANDELBROT_DATA_ROOT")) {
            if (*environment != '\0') selected = environment;
        }
    }
    if (selected.empty()) selected = project_root / "work" / "mandelbrot";
    if (selected.is_relative()) selected = project_root / selected;
    return canonical_or_absolute(selected);
}

inline fs::path find_code_root(fs::path start) {
    if (start.empty() || !start.has_parent_path()) {
        std::error_code error;
        const fs::path process_executable = fs::read_symlink("/proc/self/exe", error);
        if (error || process_executable.empty()) {
            throw std::runtime_error(
                "Could not resolve the running executable; invoke the tool by path");
        }
        start = process_executable;
    }
    start = canonical_or_absolute(start);
    if (fs::is_regular_file(start)) start = start.parent_path();
    for (fs::path current = start;; current = current.parent_path()) {
        if (fs::is_regular_file(current / "build.sh") && fs::is_directory(current / "components")) return current;
        if (!current.has_parent_path() || current == current.parent_path()) break;
    }
    throw std::runtime_error("Could not locate Mandelbrot code root from " + start.string());
}

inline fs::path find_project_root(const fs::path& code_root) {
    for (fs::path current = code_root;; current = current.parent_path()) {
        if (fs::exists(current / ".git") || fs::exists(current / ".root")) return current;
        if (!current.has_parent_path() || current == current.parent_path()) break;
    }
    throw std::runtime_error("Could not locate project root from " + code_root.string());
}

inline void replace_all(std::string& text, const std::string& from, const std::string& to) {
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

class RepoConfig {
public:
    RepoConfig(
        Json root,
        fs::path config_path,
        fs::path code_root,
        fs::path project_root,
        fs::path data_root
    )
        : root_(std::move(root)),
          config_path_(std::move(config_path)),
          code_root_(std::move(code_root)),
          project_root_(std::move(project_root)),
          data_root_(std::move(data_root)) {}

    static RepoConfig load(
        const fs::path& config_arg,
        const fs::path& start,
        const fs::path& explicit_data_root = {}
    ) {
        const fs::path code_root = find_code_root(start);
        const fs::path project_root = find_project_root(code_root);
        const fs::path data_root = resolve_data_root(explicit_data_root, project_root);
        fs::path config_path = config_arg.empty() ? code_root / "mandelbrot.json" : config_arg;
        if (config_path.is_relative()) config_path = code_root / config_path;
        config_path = canonical_or_absolute(config_path);
        std::ifstream input(config_path);
        if (!input) throw std::runtime_error("Could not open repository config: " + config_path.string());
        std::ostringstream buffer; buffer << input.rdbuf();
        Json root = Json::parse(buffer.str());
        if (!root.is_object()) throw std::runtime_error("Repository config must be a JSON object");
        return RepoConfig(std::move(root), config_path, code_root, project_root, data_root);
    }

    const Json* find(const std::string& dotted) const {
        const Json* current = &root_;
        std::size_t begin = 0;
        while (begin <= dotted.size()) {
            const std::size_t end = dotted.find('.', begin);
            const std::string part = dotted.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            current = current->find(part);
            if (!current) return nullptr;
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return current;
    }

    const Json& require(const std::string& dotted) const {
        const Json* value = find(dotted);
        if (!value) throw std::runtime_error("Missing repository config value: " + dotted);
        return *value;
    }

    std::string string(const std::string& dotted, const std::string& fallback = "") const {
        const Json* value = find(dotted);
        if (!value) return fallback;
        if (!value->is_string()) throw std::runtime_error("Config value is not a string: " + dotted);
        return value->string();
    }
    long double number(const std::string& dotted, long double fallback = 0) const {
        const Json* value = find(dotted);
        if (!value) return fallback;
        if (!value->is_number()) throw std::runtime_error("Config value is not numeric: " + dotted);
        return value->number();
    }
    int integer(const std::string& dotted, int fallback = 0) const { return static_cast<int>(number(dotted, fallback)); }
    std::uint64_t u64(const std::string& dotted, std::uint64_t fallback = 0) const { return static_cast<std::uint64_t>(number(dotted, static_cast<long double>(fallback))); }
    bool boolean(const std::string& dotted, bool fallback = false) const {
        const Json* value = find(dotted);
        if (!value) return fallback;
        if (!value->is_bool()) throw std::runtime_error("Config value is not boolean: " + dotted);
        return value->boolean();
    }
    std::vector<long double> number_array(const std::string& dotted) const {
        const Json& value = require(dotted);
        if (!value.is_array()) throw std::runtime_error("Config value is not an array: " + dotted);
        std::vector<long double> result;
        for (const Json& item : value.array()) {
            if (!item.is_number()) throw std::runtime_error("Non-numeric item in config array: " + dotted);
            result.push_back(item.number());
        }
        return result;
    }
    std::vector<std::string> string_array(const std::string& dotted) const {
        const Json& value = require(dotted);
        if (!value.is_array()) throw std::runtime_error("Config value is not an array: " + dotted);
        std::vector<std::string> result;
        for (const Json& item : value.array()) {
            if (!item.is_string()) throw std::runtime_error("Non-string item in config array: " + dotted);
            result.push_back(item.string());
        }
        return result;
    }
    fs::path path(const std::string& dotted, const std::string& fallback = "") const {
        std::string value = string(dotted, fallback);
        replace_all(value, "${data_root}", data_root_.string());
        replace_all(value, "$data_root", data_root_.string());
        replace_all(value, "${code_root}", code_root_.string());
        replace_all(value, "$code_root", code_root_.string());
        replace_all(value, "${project_root}", project_root_.string());
        replace_all(value, "$project_root", project_root_.string());
        fs::path result(value);
        if (result.is_relative()) result = code_root_ / result;
        return canonical_or_absolute(result);
    }
    unsigned threads() const {
        const int configured = integer("runtime.threads", 0);
        return configured > 0 ? static_cast<unsigned>(configured)
                              : std::max(1u, std::thread::hardware_concurrency());
    }

    const fs::path& config_path() const { return config_path_; }
    const fs::path& code_root() const { return code_root_; }
    const fs::path& project_root() const { return project_root_; }
    const fs::path& data_root() const { return data_root_; }
    fs::path catalogue_root() const { return data_root_ / "component_catalogue"; }
    fs::path contours_root() const { return data_root_ / "G_contours"; }
    fs::path promotion_root() const { return project_root_ / "work" / "promote" / "mandelbrot"; }

private:
    Json root_;
    fs::path config_path_;
    fs::path code_root_;
    fs::path project_root_;
    fs::path data_root_;
};

struct CliConfig {
    fs::path config;
    bool help = false;
    std::vector<std::string> remaining;
};

inline CliConfig parse_common_cli(int argc, char** argv, const std::string& usage) {
    CliConfig result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            if (++i >= argc) throw std::runtime_error("--config requires a path\n" + usage);
            result.config = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            result.help = true;
        } else {
            result.remaining.push_back(arg);
        }
    }
    return result;
}

} // namespace mandelbrot::repo
