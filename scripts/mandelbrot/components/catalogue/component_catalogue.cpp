#include "component_catalogue.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <charconv>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <sqlite3.h>

namespace mandelbrot::catalogue {

class CatalogueDatabase {
public:
    explicit CatalogueDatabase(const fs::path& path) {
        fs::create_directories(path.parent_path());
        const std::string filename = path.string();
        const int result = sqlite3_open_v2(
            filename.c_str(),
            &handle_,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nullptr);
        if (result != SQLITE_OK) {
            const std::string message = handle_
                ? sqlite3_errmsg(handle_) : "unknown SQLite error";
            if (handle_) sqlite3_close(handle_);
            handle_ = nullptr;
            throw std::runtime_error(
                "Could not open SQLite catalogue " + filename + ": " + message);
        }
        sqlite3_extended_result_codes(handle_, 1);
        sqlite3_busy_timeout(handle_, 60000);
        try {
            execute("PRAGMA foreign_keys = ON;");
            execute("PRAGMA journal_mode = WAL;");
            execute("PRAGMA synchronous = FULL;");
            execute("PRAGMA wal_autocheckpoint = 1000;");

            const fs::path schema_path =
                fs::path(__FILE__).parent_path() / "schema.sql";
            std::ifstream input(schema_path);
            if (!input) {
                throw std::runtime_error(
                    "Could not open catalogue schema: " + schema_path.string());
            }
            const std::string schema{
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()};
            execute(schema);
        } catch (...) {
            sqlite3_close(handle_);
            handle_ = nullptr;
            throw;
        }
    }

    ~CatalogueDatabase() {
        if (handle_) sqlite3_close(handle_);
    }

    CatalogueDatabase(const CatalogueDatabase&) = delete;
    CatalogueDatabase& operator=(const CatalogueDatabase&) = delete;

    sqlite3* handle() const noexcept { return handle_; }

    void execute(const std::string& sql) const {
        char* raw_message = nullptr;
        const int result = sqlite3_exec(
            handle_, sql.c_str(), nullptr, nullptr, &raw_message);
        if (result != SQLITE_OK) {
            const std::string message = raw_message
                ? raw_message : sqlite3_errmsg(handle_);
            sqlite3_free(raw_message);
            throw std::runtime_error("SQLite error: " + message);
        }
    }

    mutable std::recursive_mutex mutex;

private:
    sqlite3* handle_ = nullptr;
};

namespace {

std::string sqlite_error(sqlite3* database, const std::string& operation) {
    return operation + ": " + sqlite3_errmsg(database)
        + " (SQLite code "
        + std::to_string(sqlite3_extended_errcode(database)) + ")";
}

class Statement {
public:
    Statement(sqlite3* database, const char* sql) : database_(database) {
        const int result = sqlite3_prepare_v2(
            database_, sql, -1, &statement_, nullptr);
        if (result != SQLITE_OK) {
            throw std::runtime_error(sqlite_error(
                database_, "Could not prepare catalogue statement"));
        }
    }

    ~Statement() {
        if (statement_) sqlite3_finalize(statement_);
    }

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    void bind_text(int index, const std::string& value) {
        check(sqlite3_bind_text(
            statement_, index, value.c_str(),
            static_cast<int>(value.size()), SQLITE_TRANSIENT),
            "Could not bind catalogue text");
    }

    void bind_int(int index, int value) {
        check(sqlite3_bind_int(statement_, index, value),
              "Could not bind catalogue integer");
    }

    void bind_int64(int index, std::int64_t value) {
        check(sqlite3_bind_int64(statement_, index, value),
              "Could not bind catalogue integer");
    }

    void bind_double(int index, double value) {
        check(sqlite3_bind_double(statement_, index, value),
              "Could not bind catalogue real");
    }

    void bind_null(int index) {
        check(sqlite3_bind_null(statement_, index),
              "Could not bind catalogue NULL");
    }

    int step() {
        return sqlite3_step(statement_);
    }

    void expect_done(const std::string& operation) {
        const int result = step();
        if (result != SQLITE_DONE) {
            throw std::runtime_error(sqlite_error(database_, operation));
        }
    }

    void reset() {
        check(sqlite3_reset(statement_),
              "Could not reset catalogue statement");
        check(sqlite3_clear_bindings(statement_),
              "Could not clear catalogue statement bindings");
    }

    std::string column_text(int index) const {
        const unsigned char* value = sqlite3_column_text(statement_, index);
        if (!value) return {};
        return reinterpret_cast<const char*>(value);
    }

    int column_int(int index) const {
        return sqlite3_column_int(statement_, index);
    }

    std::int64_t column_int64(int index) const {
        return sqlite3_column_int64(statement_, index);
    }

private:
    void check(int result, const std::string& operation) {
        if (result != SQLITE_OK) {
            throw std::runtime_error(sqlite_error(database_, operation));
        }
    }

    sqlite3* database_ = nullptr;
    sqlite3_stmt* statement_ = nullptr;
};

std::size_t count_rows_for_run(
    CatalogueDatabase& database,
    const char* sql,
    const std::string& run_name,
    const std::string& operation
) {
    Statement statement(database.handle(), sql);
    statement.bind_text(1, run_name);
    const int step_result = statement.step();
    if (step_result != SQLITE_ROW) {
        throw std::runtime_error(sqlite_error(database.handle(), operation));
    }
    const std::int64_t count = statement.column_int64(0);
    if (count < 0) {
        throw std::runtime_error(operation + ": SQLite returned a negative row count");
    }
    return static_cast<std::size_t>(count);
}

std::size_t count_rows_for_run_period(
    CatalogueDatabase& database,
    const char* sql,
    const std::string& run_name,
    int period,
    const std::string& operation
) {
    Statement statement(database.handle(), sql);
    statement.bind_text(1, run_name);
    statement.bind_int(2, period);
    const int step_result = statement.step();
    if (step_result != SQLITE_ROW) {
        throw std::runtime_error(sqlite_error(database.handle(), operation));
    }
    const std::int64_t count = statement.column_int64(0);
    if (count < 0) {
        throw std::runtime_error(operation + ": SQLite returned a negative row count");
    }
    return static_cast<std::size_t>(count);
}

std::size_t count_rows_for_run_period_rho(
    CatalogueDatabase& database,
    const char* sql,
    const std::string& run_name,
    int period,
    const std::string& rho,
    const std::string& operation
) {
    Statement statement(database.handle(), sql);
    statement.bind_text(1, run_name);
    statement.bind_int(2, period);
    statement.bind_text(3, rho);
    const int step_result = statement.step();
    if (step_result != SQLITE_ROW) {
        throw std::runtime_error(sqlite_error(database.handle(), operation));
    }
    const std::int64_t count = statement.column_int64(0);
    if (count < 0) {
        throw std::runtime_error(operation + ": SQLite returned a negative row count");
    }
    return static_cast<std::size_t>(count);
}

class Transaction {
public:
    explicit Transaction(CatalogueDatabase& database) : database_(database) {
        owner_ = sqlite3_get_autocommit(database_.handle()) != 0;
        if (owner_) database_.execute("BEGIN IMMEDIATE;");
    }

    ~Transaction() {
        if (owner_ && !committed_) {
            try { database_.execute("ROLLBACK;"); } catch (...) {}
        }
    }

    void commit() {
        if (owner_) database_.execute("COMMIT;");
        committed_ = true;
    }

private:
    CatalogueDatabase& database_;
    bool owner_ = false;
    bool committed_ = false;
};

struct Json;
using Object = std::map<std::string, Json>;
using Array = std::vector<Json>;
struct Json : std::variant<std::nullptr_t, bool, std::string, Array, Object> {
    using variant::variant;
};

std::string escape_json(const std::string& s) {
    std::ostringstream out;
    for (unsigned char ch : s) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << int(ch);
                } else {
                    out << char(ch);
                }
        }
    }
    return out.str();
}

void dump_json(std::ostream& out, const Json& value, int indent = 0, int level = 0) {
    const auto pad = [&](int count) {
        if (indent) out << std::string(static_cast<std::size_t>(count), ' ');
    };
    if (std::holds_alternative<std::nullptr_t>(value)) {
        out << "null";
    } else if (const auto* p = std::get_if<bool>(&value)) {
        out << (*p ? "true" : "false");
    } else if (const auto* p = std::get_if<std::string>(&value)) {
        out << '"' << escape_json(*p) << '"';
    } else if (const auto* p = std::get_if<Array>(&value)) {
        const bool inline_scalars = indent && !p->empty()
            && p->size() <= 4
            && std::all_of(p->begin(), p->end(), [](const Json& item) {
                return std::holds_alternative<std::nullptr_t>(item)
                    || std::holds_alternative<bool>(item)
                    || std::holds_alternative<std::string>(item);
            });
        out << '[';
        for (std::size_t i = 0; i < p->size(); ++i) {
            if (i) out << ',';
            if (inline_scalars) {
                if (i) out << ' ';
            } else if (indent) {
                out << '\n';
                pad((level + 1) * indent);
            }
            dump_json(out, (*p)[i], indent, level + 1);
        }
        if (indent && !p->empty() && !inline_scalars) {
            out << '\n';
            pad(level * indent);
        }
        out << ']';
    } else {
        const auto& object = std::get<Object>(value);
        out << '{';
        std::size_t index = 0;
        for (const auto& [key, item] : object) {
            if (index++) out << ',';
            if (indent) { out << '\n'; pad((level + 1) * indent); }
            out << '"' << escape_json(key) << "\":";
            if (indent) out << ' ';
            dump_json(out, item, indent, level + 1);
        }
        if (indent && !object.empty()) { out << '\n'; pad(level * indent); }
        out << '}';
    }
}

std::string json_string(const Json& value) {
    std::ostringstream output;
    dump_json(output, value);
    return output.str();
}

class Parser {
public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Json parse() {
        skip();
        Json result = value();
        skip();
        if (pos_ != text_.size()) fail("trailing content");
        return result;
    }
private:
    std::string text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON parse error at " + std::to_string(pos_) +
                                 ": " + message);
    }
    void skip() {
        while (pos_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
    bool take(char value) {
        if (peek() == value) { ++pos_; return true; }
        return false;
    }
    Json value() {
        skip();
        const char current = peek();
        if (current == '{') return object();
        if (current == '[') return array();
        if (current == '"') return string();
        if (current == 't' && text_.substr(pos_, 4) == "true") {
            pos_ += 4; return true;
        }
        if (current == 'f' && text_.substr(pos_, 5) == "false") {
            pos_ += 5; return false;
        }
        if (current == 'n' && text_.substr(pos_, 4) == "null") {
            pos_ += 4; return nullptr;
        }
        return number_token();
    }
    Json object() {
        take('{'); skip(); Object result;
        if (take('}')) return result;
        while (true) {
            skip();
            if (peek() != '"') fail("expected object key");
            std::string key = std::get<std::string>(string());
            skip();
            if (!take(':')) fail("expected ':'");
            result.emplace(std::move(key), value());
            skip();
            if (take('}')) break;
            if (!take(',')) fail("expected ','");
        }
        return result;
    }
    Json array() {
        take('['); skip(); Array result;
        if (take(']')) return result;
        while (true) {
            result.push_back(value());
            skip();
            if (take(']')) break;
            if (!take(',')) fail("expected ','");
        }
        return result;
    }
    Json string() {
        if (!take('"')) fail("expected string");
        std::string result;
        while (pos_ < text_.size()) {
            const char current = text_[pos_++];
            if (current == '"') return result;
            if (current != '\\') { result.push_back(current); continue; }
            if (pos_ >= text_.size()) fail("bad escape");
            const char escaped = text_[pos_++];
            switch (escaped) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                default: fail("unsupported escape");
            }
        }
        fail("unterminated string");
    }
    Json number_token() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        if (peek() == '.') {
            ++pos_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (start == pos_) fail("expected value");
        // Legacy v1 JSON numbers are retained as their original token text.
        return text_.substr(start, pos_ - start);
    }
};

Json parse_json_text(const std::string& value) {
    return Parser(value).parse();
}

const Object& as_object(const Json& value) {
    if (const auto* pointer = std::get_if<Object>(&value)) return *pointer;
    throw std::runtime_error("Expected JSON object");
}
const Array& as_array(const Json& value) {
    if (const auto* pointer = std::get_if<Array>(&value)) return *pointer;
    throw std::runtime_error("Expected JSON array");
}
const Json& get(const Object& object, const std::string& key) {
    const auto it = object.find(key);
    if (it == object.end()) throw std::runtime_error("Missing JSON key: " + key);
    return it->second;
}
const Json* maybe(const Object& object, const std::string& key) {
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}
std::string text(const Json& value) {
    if (const auto* pointer = std::get_if<std::string>(&value)) return *pointer;
    throw std::runtime_error("Expected JSON string/decimal token");
}
bool boolean(const Json& value) {
    if (const auto* pointer = std::get_if<bool>(&value)) return *pointer;
    throw std::runtime_error("Expected JSON bool");
}
std::optional<std::string> optional_text(const Object& object,
                                         const std::string& key) {
    const Json* value = maybe(object, key);
    if (!value || std::holds_alternative<std::nullptr_t>(*value)) return std::nullopt;
    return text(*value);
}

CatalogueReal decimal(const Json& value) {
    return Catalogue::parse_decimal(text(value));
}
std::optional<CatalogueReal> optional_decimal(const Object& object,
                                              const std::string& key) {
    const Json* value = maybe(object, key);
    if (!value || std::holds_alternative<std::nullptr_t>(*value)) return std::nullopt;
    return decimal(*value);
}
std::uint64_t unsigned_integer(const Json& value) {
    return static_cast<std::uint64_t>(std::stoull(text(value)));
}
std::size_t size_integer(const Json& value) {
    return static_cast<std::size_t>(std::stoull(text(value)));
}
int integer(const Json& value) { return std::stoi(text(value)); }

Json decimal_json(const CatalogueReal& value, int digits = 0) {
    return Catalogue::decimal_string(value, digits);
}
Json integer_json(std::uint64_t value) { return std::to_string(value); }
Json complex_json(const ComplexValue& value, int digits = 0) {
    return Array{decimal_json(value.re, digits), decimal_json(value.im, digits)};
}
ComplexValue parse_complex(const Json& value) {
    if (const auto* object = std::get_if<Object>(&value)) {
        return {decimal(get(*object, "re")), decimal(get(*object, "im"))};
    }
    // v1 compatibility: [re, im]
    const auto& array = as_array(value);
    if (array.size() != 2) throw std::runtime_error("Complex value requires two coordinates");
    return {decimal(array[0]), decimal(array[1])};
}
Json optional_json(const std::optional<std::string>& value) {
    return value ? Json(*value) : Json(nullptr);
}
Json optional_decimal_json(
    const std::optional<CatalogueReal>& value,
    int digits
) {
    return value ? decimal_json(*value, digits) : Json(nullptr);
}
Json optional_complex_json(
    const std::optional<ComplexValue>& value,
    int digits = 0
) {
    return value ? complex_json(*value, digits) : Json(nullptr);
}

Json component_to_json(const ComponentRecord& component) {
    const int digits = component.numeric.working_precision_digits > 0
        ? component.numeric.working_precision_digits
        : 0;
    Array polygon;
    polygon.reserve(component.geometry.polygon.size());
    for (const auto& point : component.geometry.polygon) {
        polygon.push_back(complex_json(point, digits));
    }
    Array aliases;
    for (const auto& alias : component.provenance.aliases) aliases.push_back(alias);
    Array warnings;
    for (const auto& warning : component.quality.warnings) warnings.push_back(warning);

    Object hierarchy{
        {"geometricParent", optional_json(component.hierarchy.geometric_parent)},
        {"renormalizationParent", optional_json(component.hierarchy.renormalization_parent)},
        {"hierarchyRoot", optional_json(component.hierarchy.hierarchy_root)},
        {"generation", component.hierarchy.generation
            ? Json(std::to_string(*component.hierarchy.generation)) : Json(nullptr)}
    };
    if (component.hierarchy.attachment) {
        const auto& attachment = *component.hierarchy.attachment;
        hierarchy["attachment"] = Object{
            {"parentPoint", optional_complex_json(attachment.parent_point, digits)},
            {"childPointCentered", optional_complex_json(attachment.child_point_centered, digits)},
            {"gap", optional_decimal_json(attachment.gap, digits)},
            {"gapRelativeToChildSize", optional_decimal_json(attachment.gap_relative_to_child_size, digits)},
            {"verified", attachment.verified}
        };
    } else {
        hierarchy["attachment"] = nullptr;
    }

    Json circle_fit = nullptr;
    if (component.classification.circle_fit) {
        const auto& fit = *component.classification.circle_fit;
        circle_fit = Object{
            {"centerCentered", optional_complex_json(fit.center_centered, digits)},
            {"radius", optional_decimal_json(fit.radius, digits)},
            {"rms", decimal_json(fit.rms, digits)},
            {"maxError", optional_decimal_json(fit.max_error, digits)}
        };
    }
    Json cardioid_fit = nullptr;
    if (component.classification.cardioid_fit) {
        const auto& fit = *component.classification.cardioid_fit;
        cardioid_fit = Object{
            {"centerCentered", optional_complex_json(fit.center_centered, digits)},
            {"size", optional_decimal_json(fit.size, digits)},
            {"angle", decimal_json(fit.angle, digits)},
            {"xi", decimal_json(fit.xi, digits)},
            {"rms", decimal_json(fit.rms, digits)},
            {"maxError", optional_decimal_json(fit.max_error, digits)}
        };
    }

    return Object{
        {"schema", std::string(kComponentSchema)},
        {"id", component.id},
        {"numeric", Object{
            {"encoding", component.numeric.encoding},
            {"workingPrecisionDigits", std::to_string(component.numeric.working_precision_digits)},
            {"validatedDigits", std::to_string(component.numeric.validated_digits)}
        }},
        {"dynamics", Object{
            {"period", std::to_string(component.period)},
            {"center", complex_json(component.center, digits)},
            {"multiplierFamily", component.family}
        }},
        {"geometry", Object{
            {"coordinateFrame", component.geometry.coordinate_frame},
            {"polygonRho", decimal_json(component.geometry.polygon_rho, digits)},
            {"polygon", polygon},
            {"polygonArea", decimal_json(component.geometry.polygon_area, digits)},
            {"areaEstimate", decimal_json(component.geometry.area_estimate, digits)},
            {"areaError", decimal_json(component.geometry.area_error, digits)},
            {"areaRho", decimal_json(component.geometry.area_rho, digits)},
            {"characteristicSize", decimal_json(component.geometry.characteristic_size, digits)},
            {"bboxCentered", Array{
                decimal_json(component.geometry.bbox_centered[0], digits),
                decimal_json(component.geometry.bbox_centered[1], digits),
                decimal_json(component.geometry.bbox_centered[2], digits),
                decimal_json(component.geometry.bbox_centered[3], digits)
            }}
        }},
        {"classification", Object{
            {"shapeClass", component.classification.shape_class},
            {"shapeConfidence", decimal_json(component.classification.shape_confidence, digits)},
            {"circleFit", circle_fit},
            {"cardioidFit", cardioid_fit}
        }},
        {"symmetry", Object{
            {"relation", component.symmetry.relation},
            {"multiplicity", std::to_string(component.symmetry.multiplicity)}
        }},
        {"hierarchy", hierarchy},
        {"provenance", Object{
            {"method", component.provenance.method},
            {"runId", component.provenance.run_id},
            {"discoveredAt", component.provenance.discovered_at},
            {"softwareRevision", component.provenance.software_revision},
            {"aliases", aliases}
        }},
        {"quality", Object{
            {"centerValidated", component.quality.center_validated},
            {"exactPeriodValidated", component.quality.exact_period_validated},
            {"polygonConverged", component.quality.polygon_converged},
            {"areaAboveCutoff", component.quality.area_above_cutoff},
            {"warnings", warnings}
        }}
    };
}

bool component_records_equal(
    const ComponentRecord& lhs,
    const ComponentRecord& rhs) {
    return component_to_json(lhs) == component_to_json(rhs);
}

ComponentRecord component_from_json(const Json& value) {
    const auto& root = as_object(value);
    const std::string schema = text(get(root, "schema"));
    const bool legacy = schema == "mandelbrot-component-v1";
    if (!legacy && schema != "mandelbrot-component-v2"
        && schema != "mandelbrot-component-v3" && schema != kComponentSchema) {
        throw std::runtime_error("Unsupported component schema: " + schema);
    }

    ComponentRecord component;
    component.id = text(get(root, "id"));
    if (legacy) {
        const Object dynamics = as_object(get(root, "dynamics"));
        component.period = integer(get(dynamics, "period"));
        component.center = parse_complex(get(dynamics, "center"));
        component.numeric.working_precision_digits = integer(get(dynamics, "centerPrecisionDigits"));
        component.numeric.validated_digits = component.numeric.working_precision_digits;
        component.family = text(get(dynamics, "multiplierFamily"));
    } else {
        const Object numeric = as_object(get(root, "numeric"));
        component.numeric.encoding = text(get(numeric, "encoding"));
        component.numeric.working_precision_digits = integer(get(numeric, "workingPrecisionDigits"));
        component.numeric.validated_digits = integer(get(numeric, "validatedDigits"));
        const Object dynamics = as_object(get(root, "dynamics"));
        component.period = integer(get(dynamics, "period"));
        component.center = parse_complex(get(dynamics, "center"));
        component.family = text(get(dynamics, "multiplierFamily"));
    }

    const Object geometry = as_object(get(root, "geometry"));
    component.geometry.coordinate_frame = text(get(geometry, "coordinateFrame"));
    component.geometry.polygon_rho = decimal(get(geometry, "polygonRho"));
    const Array polygon_values = as_array(get(geometry, "polygon"));
    for (const auto& point : polygon_values) {
        component.geometry.polygon.push_back(parse_complex(point));
    }
    component.geometry.polygon_area = decimal(get(geometry, "polygonArea"));
    component.geometry.area_estimate = decimal(get(geometry, "areaEstimate"));
    component.geometry.area_error = decimal(get(geometry, "areaError"));
    component.geometry.area_rho = decimal(get(geometry, "areaRho"));
    component.geometry.characteristic_size = decimal(get(geometry, "characteristicSize"));
    const Array bounds = as_array(get(geometry, "bboxCentered"));
    if (bounds.size() != 4) throw std::runtime_error("bboxCentered requires four values");
    for (std::size_t i = 0; i < 4; ++i) component.geometry.bbox_centered[i] = decimal(bounds[i]);

    const Object classification = as_object(get(root, "classification"));
    component.classification.shape_class = text(get(classification, "shapeClass"));
    component.classification.shape_confidence = decimal(get(classification, "shapeConfidence"));
    if (const Json* raw = maybe(classification, "circleFit");
        raw && !std::holds_alternative<std::nullptr_t>(*raw)) {
        const auto& object = as_object(*raw);
        CircleFitRecord fit;
        if (const Json* value = maybe(object, "centerCentered");
            value && !std::holds_alternative<std::nullptr_t>(*value)) {
            fit.center_centered = parse_complex(*value);
        }
        fit.radius = optional_decimal(object, "radius");
        fit.rms = decimal(get(object, "rms"));
        fit.max_error = optional_decimal(object, "maxError");
        component.classification.circle_fit = fit;
    } else if (const auto legacy_rms = optional_decimal(classification, "circleFitRms")) {
        CircleFitRecord fit;
        fit.rms = *legacy_rms;
        component.classification.circle_fit = fit;
    }
    if (const Json* raw = maybe(classification, "cardioidFit");
        raw && !std::holds_alternative<std::nullptr_t>(*raw)) {
        const auto& object = as_object(*raw);
        CardioidFitRecord fit;
        if (const Json* value = maybe(object, "centerCentered");
            value && !std::holds_alternative<std::nullptr_t>(*value)) {
            fit.center_centered = parse_complex(*value);
        }
        fit.size = optional_decimal(object, "size");
        fit.angle = decimal(get(object, "angle"));
        fit.xi = optional_decimal(object, "xi").value_or(CatalogueReal(0));
        fit.rms = decimal(get(object, "rms"));
        fit.max_error = optional_decimal(object, "maxError");
        component.classification.cardioid_fit = fit;
    } else if (const auto legacy_rms = optional_decimal(classification, "cardioidFitRms")) {
        CardioidFitRecord fit;
        fit.rms = *legacy_rms;
        component.classification.cardioid_fit = fit;
    }
    if (component.classification.shape_class == "circle") {
        const auto& fit = component.classification.circle_fit;
        component.classification.shape_class =
            (fit && fit->center_centered && fit->radius) ? "disk" : "unknown";
    }

    const Object symmetry = as_object(get(root, "symmetry"));
    component.symmetry.relation = text(get(symmetry, "relation"));
    component.symmetry.multiplicity = integer(get(symmetry, "multiplicity"));

    const Object hierarchy = as_object(get(root, "hierarchy"));
    component.hierarchy.geometric_parent = optional_text(hierarchy, "geometricParent");
    component.hierarchy.renormalization_parent = optional_text(hierarchy, "renormalizationParent");
    component.hierarchy.hierarchy_root = optional_text(hierarchy, "hierarchyRoot");
    if (const Json* generation = maybe(hierarchy, "generation");
        generation && !std::holds_alternative<std::nullptr_t>(*generation)) {
        component.hierarchy.generation = integer(*generation);
    }
    if (const Json* attachment = maybe(hierarchy, "attachment");
        attachment && !std::holds_alternative<std::nullptr_t>(*attachment)) {
        const auto& object = as_object(*attachment);
        AttachmentRecord record;
        if (const Json* p = maybe(object, "parentPoint"); p && !std::holds_alternative<std::nullptr_t>(*p)) {
            record.parent_point = parse_complex(*p);
        }
        if (const Json* p = maybe(object, "childPointCentered"); p && !std::holds_alternative<std::nullptr_t>(*p)) {
            record.child_point_centered = parse_complex(*p);
        }
        record.gap = optional_decimal(object, "gap");
        record.gap_relative_to_child_size = optional_decimal(object, "gapRelativeToChildSize");
        record.verified = boolean(get(object, "verified"));
        component.hierarchy.attachment = record;
    }

    const Object provenance = as_object(get(root, "provenance"));
    component.provenance.method = text(get(provenance, "method"));
    component.provenance.run_id = text(get(provenance, "runId"));
    component.provenance.discovered_at = text(get(provenance, "discoveredAt"));
    component.provenance.software_revision = text(get(provenance, "softwareRevision"));
    const Array alias_values = as_array(get(provenance, "aliases"));
    for (const auto& alias : alias_values) {
        component.provenance.aliases.push_back(text(alias));
    }

    const Object quality = as_object(get(root, "quality"));
    component.quality.center_validated = boolean(get(quality, "centerValidated"));
    component.quality.exact_period_validated = boolean(get(quality, "exactPeriodValidated"));
    component.quality.polygon_converged = boolean(get(quality, "polygonConverged"));
    component.quality.area_above_cutoff = boolean(get(quality, "areaAboveCutoff"));
    const Array warning_values = as_array(get(quality, "warnings"));
    for (const auto& warning : warning_values) {
        component.quality.warnings.push_back(text(warning));
    }
    return component;
}

std::string read_all(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Could not open " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
void atomic_write(const fs::path& path, const Json& value) {
    fs::create_directories(path.parent_path());
    fs::path temporary = path;
    temporary += ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("Could not write " + temporary.string());
    dump_json(output, value, 2);
    output << '\n';
    output.flush();
    if (!output) throw std::runtime_error("Write failed: " + temporary.string());
    output.close();
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(path, error);
        error.clear();
        fs::rename(temporary, path, error);
        if (error) throw std::runtime_error("Atomic replace failed: " + error.message());
    }
}
Json load_json(const fs::path& path) { return Parser(read_all(path)).parse(); }

Json manifest_to_json(const Manifest& manifest) {
    return Object{
        {"schema", std::string(kManifestSchema)},
        {"catalogueRevision", integer_json(manifest.catalogue_revision)},
        {"family", manifest.family},
        {"canonicalHalfPlane", manifest.canonical_half_plane},
        {"componentCountStored", integer_json(manifest.component_count_stored)},
        {"componentCountWithSymmetry", integer_json(manifest.component_count_with_symmetry)},
        {"minimumArea", decimal_json(manifest.minimum_area)},
        {"exactThroughPeriod", std::to_string(manifest.exact_through_period)},
        {"createdAt", manifest.created_at},
        {"updatedAt", manifest.updated_at},
        {"softwareRevision", manifest.software_revision},
        {"numericEncoding", std::string(kNumericEncoding)},
        {"paths", Object{
            {"database", "component_catalogue.sqlite"},
            {"runs", "runs"},
            {"exports", "exports"}
        }}
    };
}
Manifest manifest_from_json(const Json& value) {
    const auto& object = as_object(value);
    const std::string schema = text(get(object, "schema"));
    if (schema != kManifestSchema && schema != "mandelbrot-catalogue-v1") {
        throw std::runtime_error("Unsupported manifest schema: " + schema);
    }
    Manifest manifest;
    manifest.catalogue_revision = unsigned_integer(get(object, "catalogueRevision"));
    manifest.family = text(get(object, "family"));
    manifest.canonical_half_plane = text(get(object, "canonicalHalfPlane"));
    manifest.component_count_stored = size_integer(get(object, "componentCountStored"));
    manifest.component_count_with_symmetry = size_integer(get(object, "componentCountWithSymmetry"));
    manifest.minimum_area = decimal(get(object, "minimumArea"));
    manifest.exact_through_period = integer(get(object, "exactThroughPeriod"));
    manifest.created_at = text(get(object, "createdAt"));
    manifest.updated_at = text(get(object, "updatedAt"));
    manifest.software_revision = text(get(object, "softwareRevision"));
    return manifest;
}

Json period_to_json(const PeriodRecord& period) {
    Array ids;
    for (const auto& id : period.component_ids) ids.push_back(id);
    return Object{
        {"schema", std::string(kPeriodSchema)},
        {"period", std::to_string(period.period)},
        {"theoreticalComponentCount", period.theoretical_component_count},
        {"knownRepresentativeCount", integer_json(period.known_representative_count)},
        {"knownComponentCountWithSymmetry", integer_json(period.known_component_count_with_symmetry)},
        {"catalogueComplete", period.catalogue_complete},
        {"knownArea", decimal_json(period.known_area)},
        {"knownAreaError", decimal_json(period.known_area_error)},
        {"areaCutoff", decimal_json(period.area_cutoff)},
        {"exactGeometryComplete", period.exact_geometry_complete},
        {"polygonRho", decimal_json(period.polygon_rho)},
        {"areaRho", decimal_json(period.area_rho)},
        {"polygonPoints", integer_json(period.polygon_points)},
        {"componentIds", ids},
        {"generatedFromCatalogueRevision", integer_json(period.generated_from_catalogue_revision)}
    };
}
PeriodRecord period_from_json(const Json& value) {
    const auto& object = as_object(value);
    const std::string schema = text(get(object, "schema"));
    if (schema != kPeriodSchema
        && schema != "mandelbrot-period-v2"
        && schema != "mandelbrot-period-v1") {
        throw std::runtime_error("Unsupported period schema: " + schema);
    }
    PeriodRecord period;
    period.period = integer(get(object, "period"));
    period.theoretical_component_count = text(get(object, "theoreticalComponentCount"));
    period.known_representative_count = size_integer(get(object, "knownRepresentativeCount"));
    period.known_component_count_with_symmetry = size_integer(get(object, "knownComponentCountWithSymmetry"));
    period.catalogue_complete = boolean(get(object, "catalogueComplete"));
    period.known_area = decimal(get(object, "knownArea"));
    period.known_area_error = decimal(get(object, "knownAreaError"));
    period.area_cutoff = decimal(get(object, "areaCutoff"));
    if (const Json* field = maybe(object, "exactGeometryComplete")) {
        period.exact_geometry_complete = boolean(*field);
    }
    if (const Json* field = maybe(object, "polygonRho")) {
        period.polygon_rho = decimal(*field);
    }
    if (const Json* field = maybe(object, "areaRho")) {
        period.area_rho = decimal(*field);
    }
    if (const Json* field = maybe(object, "polygonPoints")) {
        period.polygon_points = size_integer(*field);
    }
    const Array component_ids = as_array(get(object, "componentIds"));
    for (const auto& id : component_ids) period.component_ids.push_back(text(id));
    period.generated_from_catalogue_revision = unsigned_integer(get(object, "generatedFromCatalogueRevision"));
    return period;
}

Json hierarchy_to_json(const HierarchyTree& tree) {
    Array nodes;
    for (const auto& node : tree.nodes) {
        Array children;
        for (const auto& child : node.children) children.push_back(child);
        nodes.push_back(Object{
            {"id", node.id},
            {"parent", optional_json(node.parent)},
            {"children", children}
        });
    }
    return Object{
        {"schema", std::string(kHierarchySchema)},
        {"root", tree.root},
        {"nodes", nodes},
        {"statistics", Object{
            {"nodeCount", integer_json(tree.node_count)},
            {"maximumKnownGeneration", std::to_string(tree.maximum_known_generation)},
            {"knownArea", decimal_json(tree.known_area)},
            {"minimumStoredArea", decimal_json(tree.minimum_stored_area)},
            {"completeAboveCutoff", tree.complete_above_cutoff}
        }},
        {"generatedFromCatalogueRevision", integer_json(tree.generated_from_catalogue_revision)}
    };
}
HierarchyTree hierarchy_from_json(const Json& value) {
    const auto& object = as_object(value);
    const std::string schema = text(get(object, "schema"));
    if (schema != kHierarchySchema && schema != "mandelbrot-hierarchy-v1") {
        throw std::runtime_error("Unsupported hierarchy schema: " + schema);
    }
    HierarchyTree tree;
    tree.root = text(get(object, "root"));
    const Array node_values = as_array(get(object, "nodes"));
    for (const auto& item : node_values) {
        const auto& node_object = as_object(item);
        HierarchyNode node;
        node.id = text(get(node_object, "id"));
        node.parent = optional_text(node_object, "parent");
        const Array child_values = as_array(get(node_object, "children"));
        for (const auto& child : child_values) node.children.push_back(text(child));
        tree.nodes.push_back(std::move(node));
    }
    const Object statistics = as_object(get(object, "statistics"));
    tree.node_count = size_integer(get(statistics, "nodeCount"));
    tree.maximum_known_generation = integer(get(statistics, "maximumKnownGeneration"));
    tree.known_area = decimal(get(statistics, "knownArea"));
    tree.minimum_stored_area = decimal(get(statistics, "minimumStoredArea"));
    tree.complete_above_cutoff = boolean(get(statistics, "completeAboveCutoff"));
    tree.generated_from_catalogue_revision = unsigned_integer(get(object, "generatedFromCatalogueRevision"));
    return tree;
}

void bind_optional_text(
    Statement& statement,
    int index,
    const std::optional<std::string>& value
) {
    if (value) statement.bind_text(index, *value);
    else statement.bind_null(index);
}

Manifest load_manifest_row(CatalogueDatabase& database) {
    Statement statement(
        database.handle(),
        "SELECT record_json FROM catalogue_manifest WHERE singleton = 1;");
    const int result = statement.step();
    if (result == SQLITE_ROW) {
        return manifest_from_json(parse_json_text(statement.column_text(0)));
    }
    if (result == SQLITE_DONE) {
        throw std::runtime_error("SQLite catalogue has no manifest");
    }
    throw std::runtime_error(sqlite_error(
        database.handle(), "Could not load catalogue manifest"));
}

void save_manifest_row(
    CatalogueDatabase& database,
    const Manifest& manifest
) {
    Statement statement(
        database.handle(),
        "INSERT INTO catalogue_manifest(singleton, record_json) VALUES(1, ?1) "
        "ON CONFLICT(singleton) DO UPDATE SET record_json = excluded.record_json;");
    statement.bind_text(1, json_string(manifest_to_json(manifest)));
    statement.expect_done("Could not save catalogue manifest");
}

void save_component_row(
    CatalogueDatabase& database,
    const ComponentRecord& component
) {
    Statement component_statement(
        database.handle(),
        "INSERT INTO components("
        "uuid, period, center_re_text, center_im_text, center_re_real, "
        "center_im_real, area_estimate_text, area_estimate_real, "
        "area_error_text, characteristic_size_real, polygon_points, "
        "multiplicity, shape_class, provenance_method, hierarchy_root_uuid, "
        "geometric_parent_uuid, center_validated, exact_period_validated, "
        "polygon_converged, area_above_cutoff"
        ") VALUES("
        "?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, "
        "?15, ?16, ?17, ?18, ?19, ?20"
        ") ON CONFLICT(uuid) DO UPDATE SET "
        "period = excluded.period, "
        "center_re_text = excluded.center_re_text, "
        "center_im_text = excluded.center_im_text, "
        "center_re_real = excluded.center_re_real, "
        "center_im_real = excluded.center_im_real, "
        "area_estimate_text = excluded.area_estimate_text, "
        "area_estimate_real = excluded.area_estimate_real, "
        "area_error_text = excluded.area_error_text, "
        "characteristic_size_real = excluded.characteristic_size_real, "
        "polygon_points = excluded.polygon_points, "
        "multiplicity = excluded.multiplicity, "
        "shape_class = excluded.shape_class, "
        "provenance_method = excluded.provenance_method, "
        "hierarchy_root_uuid = excluded.hierarchy_root_uuid, "
        "geometric_parent_uuid = excluded.geometric_parent_uuid, "
        "center_validated = excluded.center_validated, "
        "exact_period_validated = excluded.exact_period_validated, "
        "polygon_converged = excluded.polygon_converged, "
        "area_above_cutoff = excluded.area_above_cutoff, "
        "updated_at = CURRENT_TIMESTAMP;");
    component_statement.bind_text(1, component.id);
    component_statement.bind_int(2, component.period);
    component_statement.bind_text(
        3, Catalogue::decimal_string(component.center.re));
    component_statement.bind_text(
        4, Catalogue::decimal_string(component.center.im));
    component_statement.bind_double(
        5, component.center.re.convert_to<double>());
    component_statement.bind_double(
        6, component.center.im.convert_to<double>());
    component_statement.bind_text(
        7, Catalogue::decimal_string(component.geometry.area_estimate));
    component_statement.bind_double(
        8, component.geometry.area_estimate.convert_to<double>());
    component_statement.bind_text(
        9, Catalogue::decimal_string(component.geometry.area_error));
    component_statement.bind_double(
        10, component.geometry.characteristic_size.convert_to<double>());
    component_statement.bind_int(
        11, static_cast<int>(component.geometry.polygon.size()));
    component_statement.bind_int(12, component.symmetry.multiplicity);
    component_statement.bind_text(13, component.classification.shape_class);
    component_statement.bind_text(14, component.provenance.method);
    bind_optional_text(
        component_statement, 15, component.hierarchy.hierarchy_root);
    bind_optional_text(
        component_statement, 16, component.hierarchy.geometric_parent);
    component_statement.bind_int(
        17, component.quality.center_validated ? 1 : 0);
    component_statement.bind_int(
        18, component.quality.exact_period_validated ? 1 : 0);
    component_statement.bind_int(
        19, component.quality.polygon_converged ? 1 : 0);
    component_statement.bind_int(
        20, component.quality.area_above_cutoff ? 1 : 0);
    component_statement.expect_done("Could not save component index row");

    Statement payload_statement(
        database.handle(),
        "INSERT INTO component_records(component_id, record_json) "
        "SELECT id, ?2 FROM components WHERE uuid = ?1 "
        "ON CONFLICT(component_id) DO UPDATE SET "
        "record_json = excluded.record_json;");
    payload_statement.bind_text(1, component.id);
    payload_statement.bind_text(
        2, json_string(component_to_json(component)));
    payload_statement.expect_done("Could not save component payload");
}

ComponentRecord load_component_row(
    CatalogueDatabase& database,
    const std::string& id
) {
    Statement statement(
        database.handle(),
        "SELECT r.record_json "
        "FROM components AS c "
        "JOIN component_records AS r ON r.component_id = c.id "
        "WHERE c.uuid = ?1;");
    statement.bind_text(1, id);
    const int result = statement.step();
    if (result == SQLITE_ROW) {
        return component_from_json(parse_json_text(statement.column_text(0)));
    }
    if (result == SQLITE_DONE) {
        throw std::runtime_error("Unknown component ID: " + id);
    }
    throw std::runtime_error(sqlite_error(
        database.handle(), "Could not load component " + id));
}

void save_period_row(
    CatalogueDatabase& database,
    const PeriodRecord& period
) {
    Statement statement(
        database.handle(),
        "INSERT INTO period_records(period, record_json) VALUES(?1, ?2) "
        "ON CONFLICT(period) DO UPDATE SET record_json = excluded.record_json;");
    statement.bind_int(1, period.period);
    statement.bind_text(2, json_string(period_to_json(period)));
    statement.expect_done("Could not save period record");
}

PeriodRecord load_period_row(
    CatalogueDatabase& database,
    int period
) {
    Statement statement(
        database.handle(),
        "SELECT record_json FROM period_records WHERE period = ?1;");
    statement.bind_int(1, period);
    const int result = statement.step();
    if (result == SQLITE_ROW) {
        return period_from_json(parse_json_text(statement.column_text(0)));
    }
    if (result == SQLITE_DONE) {
        throw std::runtime_error(
            "Catalogue has no period record for " + std::to_string(period));
    }
    throw std::runtime_error(sqlite_error(
        database.handle(), "Could not load period record"));
}

void save_hierarchy_row(
    CatalogueDatabase& database,
    const HierarchyTree& tree
) {
    Statement statement(
        database.handle(),
        "INSERT INTO hierarchy_records(root_uuid, record_json) VALUES(?1, ?2) "
        "ON CONFLICT(root_uuid) DO UPDATE SET record_json = excluded.record_json;");
    statement.bind_text(1, tree.root);
    statement.bind_text(2, json_string(hierarchy_to_json(tree)));
    statement.expect_done("Could not save hierarchy record");
}

HierarchyTree load_hierarchy_row(
    CatalogueDatabase& database,
    const std::string& root_id
) {
    Statement statement(
        database.handle(),
        "SELECT record_json FROM hierarchy_records WHERE root_uuid = ?1;");
    statement.bind_text(1, root_id);
    const int result = statement.step();
    if (result == SQLITE_ROW) {
        return hierarchy_from_json(parse_json_text(statement.column_text(0)));
    }
    if (result == SQLITE_DONE) {
        throw std::runtime_error(
            "Catalogue has no hierarchy record for " + root_id);
    }
    throw std::runtime_error(sqlite_error(
        database.handle(), "Could not load hierarchy record"));
}

CatalogueReal distance(const ComplexValue& lhs, const ComplexValue& rhs) {
    using boost::multiprecision::sqrt;
    const CatalogueReal dx = lhs.re - rhs.re;
    const CatalogueReal dy = lhs.im - rhs.im;
    return sqrt(dx * dx + dy * dy);
}

using CsvObject = std::unordered_map<std::string, std::string>;

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted) {
            if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else if (ch == '"') {
                quoted = false;
            } else {
                field.push_back(ch);
            }
        } else if (ch == '"') {
            quoted = true;
        } else if (ch == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result = "\"";
    for (const char ch : value) {
        if (ch == '"') result += "\"\"";
        else result.push_back(ch);
    }
    result.push_back('"');
    return result;
}

std::string csv_row_text(const std::vector<std::string>& values) {
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << csv_escape(values[index]);
    }
    return output.str();
}

CsvObject csv_object_from_row(
    const std::vector<std::string>& headers,
    const std::string& line
) {
    const auto values = parse_csv_line(line);
    CsvObject row;
    for (std::size_t index = 0; index < headers.size(); ++index) {
        row.emplace(
            headers[index],
            index < values.size() ? values[index] : std::string{});
    }
    return row;
}

template <typename Function>
void for_each_csv_object(
    const fs::path& path,
    Function function,
    const AreaScanProgressCallback& progress = {}
) {
    std::ifstream input(path);
    if (!input) {
        if (progress) progress(0, 0);
        return;
    }

    std::error_code size_error;
    const std::uintmax_t file_bytes = fs::file_size(path, size_error);
    const std::size_t total_bytes = size_error
        ? 0
        : static_cast<std::size_t>(std::min<std::uintmax_t>(
              file_bytes,
              static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())));
    std::size_t processed_bytes = 0;
    std::size_t callback_counter = 0;
    if (progress) progress(0, total_bytes);

    std::string line;
    if (!std::getline(input, line)) {
        if (progress) progress(total_bytes, total_bytes);
        return;
    }
    processed_bytes += line.size() + 1;
    const auto headers = parse_csv_line(line);

    while (std::getline(input, line)) {
        processed_bytes += line.size() + 1;
        if (line.empty()) continue;
        const auto values = parse_csv_line(line);
        CsvObject row;
        for (std::size_t i = 0; i < headers.size(); ++i) {
            row.emplace(headers[i], i < values.size() ? values[i] : "");
        }

        // Invoke the decoder before advancing the bar, so displayed progress
        // includes parsing and multiprecision conversion as well as file I/O.
        function(row);
        if (progress && (++callback_counter % 1024 == 0)) {
            progress(
                total_bytes > 0 ? std::min(processed_bytes, total_bytes)
                                : processed_bytes,
                total_bytes);
        }
    }
    if (progress) {
        progress(total_bytes > 0 ? total_bytes : processed_bytes, total_bytes);
    }
}

template <typename RowBuilder>
void write_csv_rows_stream(
    const fs::path& path,
    const std::vector<std::string>& headers,
    std::size_t row_count,
    RowBuilder build_row,
    const AreaScanProgressCallback& progress = {}
) {
    fs::create_directories(path.parent_path());
    fs::path temporary = path;
    temporary += ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());

    try {
        std::ofstream output(temporary);
        if (!output) {
            throw std::runtime_error("Could not write " + temporary.string());
        }
        for (std::size_t i = 0; i < headers.size(); ++i) {
            if (i) output << ',';
            output << csv_escape(headers[i]);
        }
        output << '\n';

        if (progress) progress(0, row_count);
        for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
            const std::vector<std::string> values = build_row(row_index);
            if (values.size() != headers.size()) {
                throw std::runtime_error("CSV row has the wrong number of fields");
            }
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i) output << ',';
                output << csv_escape(values[i]);
            }
            output << '\n';
            if (progress
                && ((row_index + 1) % 1024 == 0
                    || row_index + 1 == row_count)) {
                progress(row_index + 1, row_count);
            }
        }
        output.close();
        if (!output) {
            throw std::runtime_error("Could not finish writing " + temporary.string());
        }

        std::error_code error;
        fs::rename(temporary, path, error);
        if (error) {
            fs::remove(path, error);
            error.clear();
            fs::rename(temporary, path, error);
            if (error) {
                throw std::runtime_error(
                    "Atomic replace failed: " + error.message());
            }
        }
    } catch (...) {
        std::error_code cleanup_error;
        fs::remove(temporary, cleanup_error);
        throw;
    }
}

bool parse_cache_bool(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (unsigned char ch : value) normalized.push_back(static_cast<char>(std::tolower(ch)));
    return normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on";
}

std::optional<CatalogueReal> parse_optional_decimal(const std::string& value) {
    if (value.empty() || value == "nan" || value == "NaN" || value == "null") return std::nullopt;
    return Catalogue::parse_decimal(value);
}

std::string optional_decimal_string(const std::optional<CatalogueReal>& value) {
    return value ? Catalogue::decimal_string(*value) : "nan";
}

const std::vector<std::string>& measurement_headers() {
    static const std::vector<std::string> headers{
        "period", "component_index", "conjugate_index", "symmetry_source_component_index",
        "center_re", "center_im", "rho", "theta_points", "area_polygon", "area_derivative",
        "area_fourier", "area_estimate", "method_spread", "spectral_spread", "resolution_delta",
        "error_estimate", "fourier_tail_ratio", "negative_mode_ratio", "closure_error",
        "marked_z_closure_error", "max_residual", "solve_calls", "failed_attempts",
        "newton_iterations", "max_subdivision_depth", "rejected_branch_candidates",
        "cyclic_seed_attempts", "cyclic_recoveries", "mp_solve_calls", "mp_recoveries",
        "max_mp_dps", "seed_rho", "converged", "exact_area_at_rho",
        "exact_relative_error", "failure_reason"};
    return headers;
}

AreaMeasurementRecord measurement_from_csv(const CsvObject& row) {
    AreaMeasurementRecord record;
    record.period = std::stoi(row.at("period"));
    record.component_index = std::stoi(row.at("component_index"));
    record.conjugate_index = std::stoi(row.at("conjugate_index"));
    record.symmetry_source_component_index =
        std::stoi(row.at("symmetry_source_component_index"));
    record.center = {
        Catalogue::parse_decimal(row.at("center_re")),
        Catalogue::parse_decimal(row.at("center_im"))};
    record.rho = Catalogue::parse_decimal(row.at("rho"));
    record.theta_points = std::stoi(row.at("theta_points"));
    record.area_polygon = Catalogue::parse_decimal(row.at("area_polygon"));
    record.area_derivative = Catalogue::parse_decimal(row.at("area_derivative"));
    record.area_fourier = parse_optional_decimal(row.at("area_fourier"));
    record.area_estimate = Catalogue::parse_decimal(row.at("area_estimate"));
    record.method_spread = Catalogue::parse_decimal(row.at("method_spread"));
    record.spectral_spread = parse_optional_decimal(row.at("spectral_spread"));
    record.resolution_delta = Catalogue::parse_decimal(row.at("resolution_delta"));
    record.error_estimate = Catalogue::parse_decimal(row.at("error_estimate"));
    record.fourier_tail_ratio =
        parse_optional_decimal(row.at("fourier_tail_ratio"));
    record.negative_mode_ratio =
        parse_optional_decimal(row.at("negative_mode_ratio"));
    record.closure_error = Catalogue::parse_decimal(row.at("closure_error"));
    record.marked_z_closure_error =
        Catalogue::parse_decimal(row.at("marked_z_closure_error"));
    record.max_residual = Catalogue::parse_decimal(row.at("max_residual"));
    record.solve_calls = std::stoll(row.at("solve_calls"));
    record.failed_attempts = std::stoll(row.at("failed_attempts"));
    record.newton_iterations = std::stoll(row.at("newton_iterations"));
    record.max_subdivision_depth =
        std::stoi(row.at("max_subdivision_depth"));
    record.rejected_branch_candidates =
        std::stoll(row.at("rejected_branch_candidates"));
    record.cyclic_seed_attempts =
        std::stoll(row.at("cyclic_seed_attempts"));
    record.cyclic_recoveries = std::stoll(row.at("cyclic_recoveries"));
    record.mp_solve_calls = std::stoi(row.at("mp_solve_calls"));
    record.mp_recoveries = std::stoi(row.at("mp_recoveries"));
    record.max_mp_dps = std::stoi(row.at("max_mp_dps"));
    record.seed_rho = parse_optional_decimal(row.at("seed_rho"));
    record.converged = parse_cache_bool(row.at("converged"));
    record.exact_area_at_rho =
        parse_optional_decimal(row.at("exact_area_at_rho"));
    record.exact_relative_error =
        parse_optional_decimal(row.at("exact_relative_error"));
    record.failure_reason = row.at("failure_reason");
    return record;
}

std::vector<std::string> measurement_fields(
    const AreaMeasurementRecord& record
) {
    return {
        std::to_string(record.period),
        std::to_string(record.component_index),
        std::to_string(record.conjugate_index),
        std::to_string(record.symmetry_source_component_index),
        Catalogue::decimal_string(record.center.re),
        Catalogue::decimal_string(record.center.im),
        Catalogue::decimal_string(record.rho),
        std::to_string(record.theta_points),
        Catalogue::decimal_string(record.area_polygon),
        Catalogue::decimal_string(record.area_derivative),
        optional_decimal_string(record.area_fourier),
        Catalogue::decimal_string(record.area_estimate),
        Catalogue::decimal_string(record.method_spread),
        optional_decimal_string(record.spectral_spread),
        Catalogue::decimal_string(record.resolution_delta),
        Catalogue::decimal_string(record.error_estimate),
        optional_decimal_string(record.fourier_tail_ratio),
        optional_decimal_string(record.negative_mode_ratio),
        Catalogue::decimal_string(record.closure_error),
        Catalogue::decimal_string(record.marked_z_closure_error),
        Catalogue::decimal_string(record.max_residual),
        std::to_string(record.solve_calls),
        std::to_string(record.failed_attempts),
        std::to_string(record.newton_iterations),
        std::to_string(record.max_subdivision_depth),
        std::to_string(record.rejected_branch_candidates),
        std::to_string(record.cyclic_seed_attempts),
        std::to_string(record.cyclic_recoveries),
        std::to_string(record.mp_solve_calls),
        std::to_string(record.mp_recoveries),
        std::to_string(record.max_mp_dps),
        optional_decimal_string(record.seed_rho),
        record.converged ? "True" : "False",
        optional_decimal_string(record.exact_area_at_rho),
        optional_decimal_string(record.exact_relative_error),
        record.failure_reason};
}

std::int64_t quantize_coordinate(const CatalogueReal& value, int bits) {
    if (bits < 1 || bits > 60) throw std::invalid_argument("ComponentKey bits must lie in 1..60");
    CatalogueReal scale = 1;
    for (int i = 0; i < bits; ++i) scale *= 2;
    using boost::multiprecision::floor;
    const CatalogueReal rounded = floor(value * scale + CatalogueReal("0.5"));
    return rounded.convert_to<std::int64_t>();
}

bool component_matches_query(const ComponentRecord& component, const ComponentQuery& query) {
    if (component.period < query.min_period || component.period > query.max_period) return false;
    if (query.min_area && component.geometry.area_estimate < *query.min_area) return false;
    if (query.max_area && component.geometry.area_estimate > *query.max_area) return false;
    if (query.require_polygon && component.geometry.polygon.size() < 3) return false;
    if (query.require_center_validated && !component.quality.center_validated) return false;
    if (query.require_exact_period_validated && !component.quality.exact_period_validated) return false;
    if (query.require_polygon_converged && !component.quality.polygon_converged) return false;
    if (query.provenance_method && component.provenance.method != *query.provenance_method) return false;
    if (query.hierarchy_root
        && component.hierarchy.hierarchy_root.value_or(component.id) != *query.hierarchy_root) {
        return false;
    }
    return true;
}

} // namespace

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

Catalogue::Catalogue(fs::path root)
    : root_(std::move(root)),
      database_(std::make_shared<CatalogueDatabase>(database_path())) {}
fs::path Catalogue::database_path() const {
    return root_ / "component_catalogue.sqlite";
}
fs::path Catalogue::manifest_path() const { return root_ / "manifest.json"; }
fs::path Catalogue::component_path(const std::string& id) const {
    if (id.size() < 2) throw std::invalid_argument("Component ID too short");
    return root_ / "catalogue/components" / id.substr(0, 2) / (id + ".json");
}
fs::path Catalogue::period_path(int period) const {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << period << ".json";
    return root_ / "catalogue/periods" / name.str();
}
fs::path Catalogue::hierarchy_path(const std::string& root_id) const {
    return root_ / "catalogue/hierarchies" / (root_id + ".json");
}
fs::path Catalogue::runs_path() const { return root_ / "runs"; }
fs::path Catalogue::exports_path() const { return root_ / "exports"; }
fs::path Catalogue::indexes_path() const { return root_ / "catalogue/indexes"; }
fs::path Catalogue::export_path(const std::string& name) const {
    if (name.empty()) throw std::invalid_argument("Export name must not be empty");
    return exports_path() / name;
}
fs::path Catalogue::run_path(
    const std::string& algorithm,
    const std::string& run_name,
    const std::string& name) const {
    if (algorithm.empty() || run_name.empty()) {
        throw std::invalid_argument("Algorithm and run name must not be empty");
    }
    fs::path path = runs_path() / algorithm / run_name;
    return name.empty() ? path : path / name;
}

void Catalogue::ensure_layout() const {
    for (const auto& path : {
            root_, runs_path(), exports_path(), root_ / "legacy"}) {
        fs::create_directories(path);
    }
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement exists(
        database_->handle(),
        "SELECT 1 FROM catalogue_manifest WHERE singleton = 1;");
    const int result = exists.step();
    if (result == SQLITE_DONE) {
        Manifest manifest;
        manifest.created_at = manifest.updated_at = utc_timestamp();
        save_manifest_row(*database_, manifest);
    } else if (result != SQLITE_ROW) {
        throw std::runtime_error(sqlite_error(
            database_->handle(), "Could not inspect catalogue manifest"));
    }
    transaction.commit();
}
Manifest Catalogue::load_manifest() const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    return load_manifest_row(*database_);
}
void Catalogue::save_manifest(const Manifest& manifest) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    save_manifest_row(*database_, manifest);
    transaction.commit();
}
ComponentRecord Catalogue::load_component(const std::string& id) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    return load_component_row(*database_, id);
}
bool Catalogue::component_exists(const std::string& id) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(),
        "SELECT 1 FROM components WHERE uuid = ?1 LIMIT 1;");
    statement.bind_text(1, id);
    const int result = statement.step();
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    throw std::runtime_error(sqlite_error(
        database_->handle(), "Could not test component existence"));
}
void Catalogue::save_component(const ComponentRecord& raw, bool bump_revision) const {
    ensure_layout();
    ComponentRecord component = canonicalize_symmetry(raw);
    validate_component(component);
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    save_component_row(*database_, component);
    if (bump_revision) {
        Manifest manifest = load_manifest_row(*database_);
        ++manifest.catalogue_revision;
        manifest.updated_at = utc_timestamp();
        save_manifest_row(*database_, manifest);
    }
    transaction.commit();
}
void Catalogue::save_components(
    const std::vector<ComponentRecord>& records,
    bool bump_revision) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    for (const auto& raw : records) {
        ComponentRecord component = canonicalize_symmetry(raw);
        validate_component(component);
        save_component_row(*database_, component);
    }
    if (bump_revision && !records.empty()) {
        Manifest manifest = load_manifest_row(*database_);
        ++manifest.catalogue_revision;
        manifest.updated_at = utc_timestamp();
        save_manifest_row(*database_, manifest);
    }
    transaction.commit();
}
void Catalogue::delete_component(const std::string& id, bool bump_revision) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement statement(
        database_->handle(), "DELETE FROM components WHERE uuid = ?1;");
    statement.bind_text(1, id);
    statement.expect_done("Could not delete component " + id);
    const bool removed = sqlite3_changes(database_->handle()) != 0;
    if (removed) {
        if (bump_revision) {
            Manifest manifest = load_manifest_row(*database_);
            ++manifest.catalogue_revision;
            manifest.updated_at = utc_timestamp();
            save_manifest_row(*database_, manifest);
        }
    }
    transaction.commit();
}
std::vector<std::string> Catalogue::list_component_ids() const {
    std::vector<std::string> ids;
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(), "SELECT uuid FROM components ORDER BY uuid;");
    while (true) {
        const int result = statement.step();
        if (result == SQLITE_DONE) break;
        if (result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not list component IDs"));
        }
        ids.push_back(statement.column_text(0));
    }
    return ids;
}
std::vector<ComponentRecord> Catalogue::query_components(
    const ComponentQuery& query,
    const AreaScanProgressCallback& progress) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    std::vector<ComponentRecord> result;
    std::string sql =
        "SELECT r.record_json FROM components AS c "
        "JOIN component_records AS r ON r.component_id = c.id "
        "WHERE c.period >= ?1 AND c.period <= ?2";
    int next_parameter = 3;
    if (query.require_polygon) sql += " AND c.polygon_points >= 3";
    if (query.require_center_validated) sql += " AND c.center_validated = 1";
    if (query.require_exact_period_validated) {
        sql += " AND c.exact_period_validated = 1";
    }
    if (query.require_polygon_converged) sql += " AND c.polygon_converged = 1";
    const int provenance_parameter = query.provenance_method
        ? next_parameter++ : 0;
    if (provenance_parameter) {
        sql += " AND c.provenance_method = ?"
            + std::to_string(provenance_parameter);
    }
    const int hierarchy_parameter = query.hierarchy_root
        ? next_parameter++ : 0;
    if (hierarchy_parameter) {
        sql += " AND COALESCE(c.hierarchy_root_uuid, c.uuid) = ?"
            + std::to_string(hierarchy_parameter);
    }
    sql += " ORDER BY c.period, c.center_re_real, c.center_im_real, c.uuid;";

    Statement statement(database_->handle(), sql.c_str());
    statement.bind_int(1, query.min_period);
    statement.bind_int(2, query.max_period);
    if (provenance_parameter) {
        statement.bind_text(provenance_parameter, *query.provenance_method);
    }
    if (hierarchy_parameter) {
        statement.bind_text(hierarchy_parameter, *query.hierarchy_root);
    }

    std::vector<std::string> payloads;
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not query catalogue components"));
        }
        payloads.push_back(statement.column_text(0));
    }

    const std::size_t progress_total = payloads.size() + 2;
    if (progress) progress(0, progress_total);
    if (progress) progress(1, progress_total);
    result.reserve(payloads.size());
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        ComponentRecord component =
            component_from_json(parse_json_text(payloads[index]));
        // Decimal area filters are confirmed after parsing; REAL accelerator
        // columns are never allowed to make a precision-sensitive decision.
        if (component_matches_query(component, query)) {
            result.push_back(std::move(component));
        }
        if (progress
            && ((index + 1) % 64 == 0 || index + 1 == payloads.size())) {
            progress(index + 2, progress_total);
        }
    }
    std::sort(result.begin(), result.end(), [](const ComponentRecord& a, const ComponentRecord& b) {
        if (a.period != b.period) return a.period < b.period;
        if (a.center.re != b.center.re) return a.center.re < b.center.re;
        if (a.center.im != b.center.im) return a.center.im < b.center.im;
        return a.id < b.id;
    });
    if (progress) progress(progress_total, progress_total);
    return result;
}

CatalogueSnapshot Catalogue::load_snapshot(const ComponentQuery& query) const {
    CatalogueSnapshot snapshot;
    snapshot.manifest = load_manifest();
    for (const int period : list_periods()) {
        if (period < query.min_period || period > query.max_period) continue;
        try { snapshot.periods.push_back(load_period(period)); } catch (...) {}
    }
    snapshot.components = query_components(query);
    for (std::size_t i = 0; i < snapshot.components.size(); ++i) {
        const auto& component = snapshot.components[i];
        snapshot.by_id.emplace(component.id, i);
        snapshot.by_key[ComponentKey::from_center(component.period, component.center)].push_back(i);
        snapshot.by_period[component.period].push_back(i);
    }
    return snapshot;
}

std::vector<ComponentRecord> Catalogue::load_components_for_period(int period) const {
    ComponentQuery query;
    query.min_period = query.max_period = period;
    return query_components(query);
}
std::optional<ComponentRecord> Catalogue::find_near_center(
    int period, ComplexValue center, const CatalogueReal& tolerance) const {
    const auto components = load_components_for_period(period);
    for (const auto& component : components) {
        if (distance(component.center, center) <= tolerance) return component;
    }
    return std::nullopt;
}
PeriodRecord Catalogue::load_period(int period) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    return load_period_row(*database_, period);
}
void Catalogue::save_period(const PeriodRecord& period) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    save_period_row(*database_, period);
    transaction.commit();
}
bool Catalogue::period_exists(int period) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(),
        "SELECT 1 FROM period_records WHERE period = ?1;");
    statement.bind_int(1, period);
    const int result = statement.step();
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    throw std::runtime_error(sqlite_error(
        database_->handle(), "Could not test period-record existence"));
}
std::vector<int> Catalogue::list_periods() const {
    std::vector<int> periods;
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(),
        "SELECT period FROM period_records ORDER BY period;");
    while (true) {
        const int result = statement.step();
        if (result == SQLITE_DONE) break;
        if (result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not list period records"));
        }
        periods.push_back(statement.column_int(0));
    }
    return periods;
}
HierarchyTree Catalogue::load_hierarchy(const std::string& root_id) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    return load_hierarchy_row(*database_, root_id);
}
void Catalogue::save_hierarchy(const HierarchyTree& tree) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    save_hierarchy_row(*database_, tree);
    transaction.commit();
}

void Catalogue::rebuild_period_indexes(const CatalogueReal& area_cutoff) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    const Manifest manifest = load_manifest();
    std::map<int, PeriodRecord> periods;
    for (const auto& id : list_component_ids()) {
        const ComponentRecord component = load_component(id);
        PeriodRecord& period = periods[component.period];
        period.period = component.period;
        period.component_ids.push_back(component.id);
        ++period.known_representative_count;
        period.known_component_count_with_symmetry += static_cast<std::size_t>(component.symmetry.multiplicity);
        period.known_area += component.geometry.area_estimate * component.symmetry.multiplicity;
        period.known_area_error += boost::multiprecision::abs(component.geometry.area_error) * component.symmetry.multiplicity;
        period.area_cutoff = area_cutoff;
        period.exact_geometry_complete = false;
        period.generated_from_catalogue_revision = manifest.catalogue_revision;
    }
    database_->execute("DELETE FROM period_records;");
    for (auto& [_, period] : periods) {
        std::sort(period.component_ids.begin(), period.component_ids.end());
        save_period(period);
    }
    transaction.commit();
}

void Catalogue::rebuild_hierarchy_indexes(const CatalogueReal& minimum_stored_area) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    const Manifest manifest = load_manifest();
    std::unordered_map<std::string, ComponentRecord> records;
    for (const auto& id : list_component_ids()) records.emplace(id, load_component(id));
    std::unordered_map<std::string, std::vector<std::string>> children;
    std::set<std::string> roots;
    for (const auto& [id, component] : records) {
        if (component.hierarchy.geometric_parent) {
            children[*component.hierarchy.geometric_parent].push_back(id);
        } else {
            roots.insert(component.hierarchy.hierarchy_root.value_or(id));
        }
    }
    database_->execute("DELETE FROM hierarchy_records;");
    for (const auto& root_id : roots) {
        if (!records.count(root_id)) continue;
        HierarchyTree tree;
        tree.root = root_id;
        tree.minimum_stored_area = minimum_stored_area;
        tree.generated_from_catalogue_revision = manifest.catalogue_revision;
        std::vector<std::string> stack{root_id};
        std::set<std::string> seen;
        while (!stack.empty()) {
            const std::string id = stack.back();
            stack.pop_back();
            if (seen.count(id) || !records.count(id)) continue;
            seen.insert(id);
            const auto& component = records.at(id);
            auto child_ids = children[id];
            std::sort(child_ids.begin(), child_ids.end());
            tree.nodes.push_back(HierarchyNode{id, component.hierarchy.geometric_parent, child_ids});
            for (auto it = child_ids.rbegin(); it != child_ids.rend(); ++it) stack.push_back(*it);
            tree.known_area += component.geometry.area_estimate * component.symmetry.multiplicity;
            tree.maximum_known_generation = std::max(tree.maximum_known_generation,
                component.hierarchy.generation.value_or(0));
        }
        tree.node_count = tree.nodes.size();
        save_hierarchy(tree);
    }
    transaction.commit();
}

void Catalogue::rebuild_manifest(int exact_through_period,
                                 const CatalogueReal& minimum_area,
                                 const std::string& software_revision) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Manifest manifest = load_manifest();
    const auto ids = list_component_ids();
    manifest.component_count_stored = ids.size();
    manifest.component_count_with_symmetry = 0;
    for (const auto& id : ids) {
        manifest.component_count_with_symmetry +=
            static_cast<std::size_t>(load_component(id).symmetry.multiplicity);
    }
    manifest.minimum_area = minimum_area;
    manifest.exact_through_period = exact_through_period;
    if (!software_revision.empty()) manifest.software_revision = software_revision;
    manifest.updated_at = utc_timestamp();
    save_manifest(manifest);
    transaction.commit();
}

std::string Catalogue::generate_uuid() {
    std::array<unsigned char, 16> bytes{};
    std::random_device random;
    for (auto& byte : bytes) byte = static_cast<unsigned char>(random());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        output << std::setw(2) << int(bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) output << '-';
    }
    return output.str();
}

std::string Catalogue::stable_id(const std::string& identity) {
    auto fnv = [](const std::string& text, std::uint64_t seed) {
        std::uint64_t hash = seed;
        for (const unsigned char ch : text) {
            hash ^= ch;
            hash *= 1099511628211ULL;
        }
        return hash;
    };
    const std::uint64_t high = fnv(identity, 14695981039346656037ULL);
    const std::uint64_t low = fnv(std::string("mandelbrot-component:") + identity,
                                  1099511628211ULL);
    std::array<unsigned char, 16> bytes{};
    for (int i = 0; i < 8; ++i) {
        bytes[static_cast<std::size_t>(i)] = static_cast<unsigned char>(
            (high >> (56 - 8 * i)) & 0xffU);
        bytes[static_cast<std::size_t>(8 + i)] = static_cast<unsigned char>(
            (low >> (56 - 8 * i)) & 0xffU);
    }
    // RFC 4122 layout. This is a deterministic, repository-local UUID using
    // the same stable FNV-derived 128 bits in C++ and Python.
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x50U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) out << '-';
    }
    return out.str();
}

ComponentRecord Catalogue::canonicalize_symmetry(
    ComponentRecord component, const CatalogueReal& tolerance) {
    using boost::multiprecision::abs;
    if (component.classification.shape_class == "circle") {
        const auto& fit = component.classification.circle_fit;
        component.classification.shape_class =
            (fit && fit->center_centered && fit->radius) ? "disk" : "unknown";
    }
    if (component.center.im < -tolerance) {
        component.center.im = -component.center.im;
        for (auto& point : component.geometry.polygon) point.im = -point.im;
        std::reverse(component.geometry.polygon.begin(), component.geometry.polygon.end());
        if (component.classification.circle_fit
            && component.classification.circle_fit->center_centered) {
            component.classification.circle_fit->center_centered->im =
                -component.classification.circle_fit->center_centered->im;
        }
        if (component.classification.cardioid_fit) {
            auto& fit = *component.classification.cardioid_fit;
            if (fit.center_centered) fit.center_centered->im = -fit.center_centered->im;
            fit.angle = -fit.angle;
            fit.xi = -fit.xi;
        }
        if (component.hierarchy.attachment) {
            auto& attachment = *component.hierarchy.attachment;
            if (attachment.parent_point) attachment.parent_point->im = -attachment.parent_point->im;
            if (attachment.child_point_centered) attachment.child_point_centered->im = -attachment.child_point_centered->im;
        }
    }
    if (abs(component.center.im) <= tolerance) {
        component.center.im = 0;
        component.symmetry.relation = "real-axis";
        component.symmetry.multiplicity = 1;
    } else {
        component.symmetry.relation = "has-conjugate";
        component.symmetry.multiplicity = 2;
    }
    return component;
}

void Catalogue::validate_component(const ComponentRecord& component) {
    if (component.id.size() < 2) throw std::invalid_argument("Component ID must contain at least two characters");
    if (component.period <= 0) throw std::invalid_argument("Component period must be positive");
    if (component.numeric.encoding != kNumericEncoding) throw std::invalid_argument("Canonical numeric encoding must be decimal-string");
    if (component.numeric.working_precision_digits < 0 || component.numeric.validated_digits < 0) {
        throw std::invalid_argument("Precision digits must be non-negative");
    }
    if (component.numeric.validated_digits > component.numeric.working_precision_digits &&
        component.numeric.working_precision_digits != 0) {
        throw std::invalid_argument("validatedDigits cannot exceed workingPrecisionDigits");
    }
    if (component.geometry.coordinate_frame != "centered") throw std::invalid_argument("Only centered polygons are canonical");
    if (component.geometry.polygon.size() < 3) throw std::invalid_argument("Component polygon requires at least three points");
    if (component.center.im < 0) throw std::invalid_argument("Canonical component center must lie in upper half-plane");
    if (component.symmetry.relation == "real-axis" && component.symmetry.multiplicity != 1) {
        throw std::invalid_argument("real-axis component multiplicity must be 1");
    }
    if (component.symmetry.relation == "has-conjugate" && component.symmetry.multiplicity != 2) {
        throw std::invalid_argument("has-conjugate component multiplicity must be 2");
    }
    const auto& classification = component.classification;
    if (classification.shape_class != "unknown"
        && classification.shape_class != "disk"
        && classification.shape_class != "cardioid") {
        throw std::invalid_argument("shapeClass must be unknown, disk, or cardioid");
    }
    if (classification.shape_confidence < 0 || classification.shape_confidence > 1) {
        throw std::invalid_argument("shapeConfidence must lie in [0,1]");
    }
    if (classification.circle_fit) {
        const auto& fit = *classification.circle_fit;
        if (fit.radius && *fit.radius <= 0) {
            throw std::invalid_argument("Circle fit radius must be positive");
        }
        if (fit.rms < 0 || (fit.max_error && *fit.max_error < 0)) {
            throw std::invalid_argument("Circle fit errors must be non-negative");
        }
    }
    if (classification.cardioid_fit) {
        const auto& fit = *classification.cardioid_fit;
        if (fit.size && *fit.size <= 0) {
            throw std::invalid_argument("Cardioid fit size must be positive");
        }
        if (boost::multiprecision::abs(fit.xi) > CatalogueReal("0.5")) {
            throw std::invalid_argument("Cardioid slant xi must lie in [-0.5,0.5]");
        }
        if (fit.rms < 0 || (fit.max_error && *fit.max_error < 0)) {
            throw std::invalid_argument("Cardioid fit errors must be non-negative");
        }
    }
    if (classification.shape_class == "disk") {
        if (!classification.circle_fit
            || !classification.circle_fit->center_centered
            || !classification.circle_fit->radius) {
            throw std::invalid_argument(
                "Disk classification requires a fitted geometric centre and radius");
        }
    }
    if (classification.shape_class == "cardioid") {
        if (!classification.cardioid_fit
            || !classification.cardioid_fit->center_centered
            || !classification.cardioid_fit->size) {
            throw std::invalid_argument(
                "Cardioid classification requires a fitted centre and size");
        }
    }
}


ComponentKey ComponentKey::from_center(
    int period,
    const ComplexValue& raw_center,
    int bits) {
    ComplexValue center = raw_center;
    if (center.im < 0) center.im = -center.im;
    return ComponentKey{
        period,
        quantize_coordinate(center.re, bits),
        quantize_coordinate(center.im, bits)};
}

std::size_t ComponentKeyHash::operator()(const ComponentKey& key) const noexcept {
    std::size_t seed = std::hash<int>{}(key.period);
    const auto mix = [&](std::size_t value) {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    mix(std::hash<std::int64_t>{}(key.center_re));
    mix(std::hash<std::int64_t>{}(key.center_im));
    return seed;
}

const ComponentRecord* CatalogueSnapshot::find_id(const std::string& id) const {
    const auto found = by_id.find(id);
    return found == by_id.end() ? nullptr : &components[found->second];
}

const ComponentRecord* CatalogueSnapshot::find_near_center(
    int period,
    const ComplexValue& center,
    const CatalogueReal& tolerance) const {
    const ComponentKey key = ComponentKey::from_center(period, center);
    if (const auto exact = by_key.find(key); exact != by_key.end()) {
        for (const std::size_t index : exact->second) {
            if (distance(components[index].center, center) <= tolerance) {
                return &components[index];
            }
        }
    }
    const auto period_found = by_period.find(period);
    if (period_found == by_period.end()) return nullptr;
    for (const std::size_t index : period_found->second) {
        if (distance(components[index].center, center) <= tolerance) {
            return &components[index];
        }
    }
    return nullptr;
}

std::vector<std::reference_wrapper<const ComponentRecord>>
CatalogueSnapshot::period_components(int period) const {
    std::vector<std::reference_wrapper<const ComponentRecord>> result;
    const auto found = by_period.find(period);
    if (found == by_period.end()) return result;
    result.reserve(found->second.size());
    for (const std::size_t index : found->second) result.emplace_back(components[index]);
    return result;
}

std::vector<ComplexValue> absolute_polygon(const ComponentRecord& component) {
    std::vector<ComplexValue> result;
    result.reserve(component.geometry.polygon.size());
    for (const auto& point : component.geometry.polygon) {
        result.push_back({component.center.re + point.re, component.center.im + point.im});
    }
    return result;
}

namespace {

int geometry_source_priority(const std::string& method) {
    if (method == "exact-period-area-scan") return 100;
    if (method == "boundary-hunter") return 70;
    if (method == "satellite-hunter") return 60;
    if (method == "quadtree-hunter") return 50;
    return method.empty() ? 0 : 10;
}

bool should_replace_geometry(
    const ComponentRecord& existing,
    const ComponentRecord& incoming) {
    const bool existing_valid = existing.quality.polygon_converged
        && existing.geometry.polygon.size() >= 3;
    const bool incoming_valid = incoming.quality.polygon_converged
        && incoming.geometry.polygon.size() >= 3;
    if (!incoming_valid) return false;
    if (!existing_valid) return true;

    const int existing_priority = geometry_source_priority(existing.provenance.method);
    const int incoming_priority = geometry_source_priority(incoming.provenance.method);
    if (incoming_priority != existing_priority) {
        return incoming_priority > existing_priority;
    }

    // Re-running the same algorithm is an authoritative refresh of its own
    // record.  Different algorithms at the same tier still need to prove that
    // they carry more validated information.
    if (!incoming.provenance.method.empty()
        && incoming.provenance.method == existing.provenance.method) {
        return incoming.numeric.validated_digits
            >= existing.numeric.validated_digits;
    }
    if (incoming.numeric.validated_digits != existing.numeric.validated_digits) {
        return incoming.numeric.validated_digits
            > existing.numeric.validated_digits;
    }
    if (incoming.numeric.working_precision_digits
        != existing.numeric.working_precision_digits) {
        return incoming.numeric.working_precision_digits
            > existing.numeric.working_precision_digits;
    }
    return incoming.geometry.polygon.size() > existing.geometry.polygon.size();
}

} // namespace

ComponentRecord Catalogue::merge_component_records(
    const ComponentRecord& existing,
    const ComponentRecord& incoming) {
    if (existing.period != incoming.period) {
        throw std::invalid_argument("Cannot merge component records with different periods");
    }
    ComponentRecord result = existing;
    const bool incoming_geometry_better = should_replace_geometry(existing, incoming);
    if (incoming_geometry_better) {
        result.center = incoming.center;
        result.numeric = incoming.numeric;
        result.geometry = incoming.geometry;
        if (!incoming.provenance.method.empty()) {
            result.provenance.method = incoming.provenance.method;
            result.provenance.run_id = incoming.provenance.run_id;
            result.provenance.discovered_at = incoming.provenance.discovered_at;
            result.provenance.software_revision = incoming.provenance.software_revision;
        }
    } else {
        if (result.geometry.area_estimate == 0 && incoming.geometry.area_estimate != 0) {
            result.geometry.area_estimate = incoming.geometry.area_estimate;
            result.geometry.area_error = incoming.geometry.area_error;
            result.geometry.area_rho = incoming.geometry.area_rho;
            result.geometry.characteristic_size = incoming.geometry.characteristic_size;
        }
        result.numeric.working_precision_digits = std::max(
            result.numeric.working_precision_digits,
            incoming.numeric.working_precision_digits);
        result.numeric.validated_digits = std::max(
            result.numeric.validated_digits,
            incoming.numeric.validated_digits);
    }
    if (result.classification.shape_class == "unknown"
        && incoming.classification.shape_class != "unknown") {
        result.classification = incoming.classification;
    }
    if (!result.hierarchy.geometric_parent && incoming.hierarchy.geometric_parent) {
        result.hierarchy.geometric_parent = incoming.hierarchy.geometric_parent;
    }
    if (!result.hierarchy.renormalization_parent && incoming.hierarchy.renormalization_parent) {
        result.hierarchy.renormalization_parent = incoming.hierarchy.renormalization_parent;
    }
    if (!result.hierarchy.hierarchy_root && incoming.hierarchy.hierarchy_root) {
        result.hierarchy.hierarchy_root = incoming.hierarchy.hierarchy_root;
    }
    if (!result.hierarchy.generation && incoming.hierarchy.generation) {
        result.hierarchy.generation = incoming.hierarchy.generation;
    }
    if (!result.hierarchy.attachment && incoming.hierarchy.attachment) {
        result.hierarchy.attachment = incoming.hierarchy.attachment;
    }
    if (result.provenance.method.empty()) result.provenance.method = incoming.provenance.method;
    if (result.provenance.run_id.empty()) result.provenance.run_id = incoming.provenance.run_id;
    if (result.provenance.discovered_at.empty()) result.provenance.discovered_at = incoming.provenance.discovered_at;
    if (result.provenance.software_revision.empty()) {
        result.provenance.software_revision = incoming.provenance.software_revision;
    }
    std::set<std::string> aliases(result.provenance.aliases.begin(), result.provenance.aliases.end());
    aliases.insert(existing.provenance.aliases.begin(), existing.provenance.aliases.end());
    aliases.insert(incoming.provenance.aliases.begin(), incoming.provenance.aliases.end());
    if (!existing.provenance.method.empty()
        && existing.provenance.method != result.provenance.method) {
        aliases.insert("discovery-method:" + existing.provenance.method);
    }
    if (!incoming.provenance.method.empty()
        && incoming.provenance.method != result.provenance.method) {
        aliases.insert("discovery-method:" + incoming.provenance.method);
    }
    result.provenance.aliases.assign(aliases.begin(), aliases.end());
    result.quality.center_validated = result.quality.center_validated || incoming.quality.center_validated;
    result.quality.exact_period_validated = result.quality.exact_period_validated || incoming.quality.exact_period_validated;
    result.quality.polygon_converged = result.quality.polygon_converged || incoming.quality.polygon_converged;
    result.quality.area_above_cutoff = result.quality.area_above_cutoff || incoming.quality.area_above_cutoff;
    std::set<std::string> warnings(result.quality.warnings.begin(), result.quality.warnings.end());
    warnings.insert(incoming.quality.warnings.begin(), incoming.quality.warnings.end());
    result.quality.warnings.assign(warnings.begin(), warnings.end());
    return result;
}

std::optional<ExactPeriodIndex> Catalogue::exact_period_index(
    const ComponentRecord& component) {
    constexpr std::string_view prefix = "period-index:";
    for (const auto& alias : component.provenance.aliases) {
        if (!alias.starts_with(prefix)) continue;
        const std::string_view payload(alias.data() + prefix.size(),
                                       alias.size() - prefix.size());
        const std::size_t separator = payload.find(':');
        if (separator == std::string_view::npos
            || payload.find(':', separator + 1) != std::string_view::npos) {
            continue;
        }
        try {
            std::size_t used_period = 0;
            std::size_t used_index = 0;
            const std::string period_text(payload.substr(0, separator));
            const std::string index_text(payload.substr(separator + 1));
            const int period = std::stoi(period_text, &used_period);
            const int component_index = std::stoi(index_text, &used_index);
            if (used_period != period_text.size()
                || used_index != index_text.size()
                || period != component.period
                || component_index < 0) {
                continue;
            }
            return ExactPeriodIndex{period, component_index};
        } catch (...) {
            // Provenance aliases are extensible. Ignore malformed aliases and
            // continue looking for a valid exact-scanner identity.
        }
    }
    return std::nullopt;
}

void Catalogue::set_exact_period_index(
    ComponentRecord& component,
    int period,
    int component_index) {
    if (period <= 0 || component_index < 0) {
        throw std::invalid_argument(
            "Exact-period identity requires a positive period and non-negative index");
    }
    if (component.period != 0 && component.period != period) {
        throw std::invalid_argument(
            "Exact-period identity period does not match ComponentRecord.period");
    }
    component.period = period;
    constexpr std::string_view prefix = "period-index:";
    auto& aliases = component.provenance.aliases;
    aliases.erase(
        std::remove_if(
            aliases.begin(), aliases.end(),
            [](const std::string& alias) { return alias.starts_with(prefix); }),
        aliases.end());
    aliases.insert(
        aliases.begin(),
        std::string(prefix) + std::to_string(period) + ":"
            + std::to_string(component_index));
}

namespace {

int key_bits_for_tolerance(const CatalogueReal& tolerance) {
    if (!(tolerance > 0)) return kComponentKeyBits;
    const long double value = tolerance.convert_to<long double>();
    if (!(value > 0) || !std::isfinite(value)) return kComponentKeyBits;
    // A cell at least twice the tolerance wide plus a 3x3 neighbor lookup
    // guarantees that all possible matches become candidates.
    const int bits = static_cast<int>(std::floor(-std::log2(2.0L * value)));
    return std::clamp(bits, 1, kComponentKeyBits);
}

std::optional<std::size_t> locate_component(
    const std::vector<ComponentRecord>& records,
    const std::unordered_map<ComponentKey, std::vector<std::size_t>, ComponentKeyHash>& index,
    int period,
    const ComplexValue& center,
    const CatalogueReal& tolerance,
    int bits) {
    const ComponentKey base = ComponentKey::from_center(period, center, bits);
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
        for (std::int64_t dy = -1; dy <= 1; ++dy) {
            const ComponentKey candidate{period, base.center_re + dx, base.center_im + dy};
            const auto found = index.find(candidate);
            if (found == index.end()) continue;
            for (const std::size_t position : found->second) {
                if (distance(records[position].center, center) <= tolerance) {
                    return position;
                }
            }
        }
    }
    return std::nullopt;
}

} // namespace

UpsertResult Catalogue::upsert_component(
    ComponentRecord component,
    const UpsertOptions& options) const {
    auto results = upsert_components({std::move(component)}, options);
    return std::move(results.front());
}

std::vector<UpsertResult> Catalogue::upsert_components(
    std::vector<ComponentRecord> components,
    const UpsertOptions& options) const {
    std::vector<UpsertResult> results;
    if (components.empty()) return results;
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);

    for (auto& component : components) {
        component = canonicalize_symmetry(std::move(component));
        if (component.id.empty()) {
            component.id = stable_id(
                "component:" + std::to_string(component.period) + ":"
                + decimal_string(component.center.re, 50) + ":"
                + decimal_string(component.center.im, 50));
        }
    }

    std::set<int> target_periods;
    for (const auto& component : components) target_periods.insert(component.period);
    std::vector<ComponentRecord> records;
    // Deduplication always consults the authoritative table.  Period records
    // are rebuildable views and may legitimately lag after a crash.
    for (const int period : target_periods) {
        auto period_records = load_components_for_period(period);
        records.reserve(records.size() + period_records.size());
        for (auto& record : period_records) {
            records.push_back(std::move(record));
        }
    }
    const int bits = key_bits_for_tolerance(options.center_tolerance);
    std::unordered_map<ComponentKey, std::vector<std::size_t>, ComponentKeyHash> index;
    index.reserve(records.size() * 2 + components.size() * 2 + 1);
    for (std::size_t position = 0; position < records.size(); ++position) {
        index[ComponentKey::from_center(
            records[position].period, records[position].center, bits)].push_back(position);
    }

    results.reserve(components.size());
    bool changed = false;
    for (auto& component : components) {
        const auto existing_position = locate_component(
            records, index, component.period, component.center,
            options.center_tolerance, bits);
        UpsertResult result;
        if (!existing_position) {
            validate_component(component);
            save_component_row(*database_, component);
            result.component = component;
            result.inserted = true;
            const std::size_t position = records.size();
            records.push_back(component);
            index[ComponentKey::from_center(
                component.period, component.center, bits)].push_back(position);
            changed = true;
        } else {
            ComponentRecord& existing = records[*existing_position];
            component.id = existing.id;
            if (options.merge_existing) {
                result.component = merge_component_records(existing, component);
                validate_component(result.component);
                if (!component_records_equal(existing, result.component)) {
                    save_component_row(*database_, result.component);
                    existing = result.component;
                    index[ComponentKey::from_center(
                        existing.period, existing.center, bits)].push_back(*existing_position);
                    result.updated = true;
                    changed = true;
                } else {
                    result.component = existing;
                }
            } else {
                result.component = existing;
            }
        }
        results.push_back(std::move(result));
    }

    if (changed && options.bump_revision) {
        Manifest manifest = load_manifest_row(*database_);
        ++manifest.catalogue_revision;
        manifest.updated_at = utc_timestamp();
        save_manifest_row(*database_, manifest);
    }
    transaction.commit();
    return results;
}

bool Catalogue::update_component_classification(
    const std::string& component_id,
    const ClassificationRecord& classification,
    bool bump_revision) const {
    return update_component_classifications(
        {ClassificationUpdate{component_id, classification}},
        bump_revision) != 0;
}

std::size_t Catalogue::update_component_classifications(
    const std::vector<ClassificationUpdate>& updates,
    bool bump_revision) const {
    if (updates.empty()) return 0;
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    std::map<std::string, ClassificationRecord> unique;
    for (const auto& update : updates) {
        if (update.component_id.empty()) {
            throw std::invalid_argument("Classification update requires a component ID");
        }
        unique[update.component_id] = update.classification;
    }

    std::size_t changed = 0;
    for (const auto& [component_id, classification] : unique) {
        ComponentRecord component = load_component(component_id);
        ComponentRecord updated = component;
        updated.classification = classification;
        updated = canonicalize_symmetry(std::move(updated));
        validate_component(updated);
        if (!component_records_equal(component, updated)) {
            save_component_row(*database_, updated);
            ++changed;
        }
    }
    if (changed && bump_revision) {
        Manifest manifest = load_manifest_row(*database_);
        ++manifest.catalogue_revision;
        manifest.updated_at = utc_timestamp();
        save_manifest_row(*database_, manifest);
    }
    transaction.commit();
    return changed;
}

void Catalogue::rebuild_period_indexes(
    const std::vector<int>& requested_periods,
    const CatalogueReal& area_cutoff) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    const Manifest manifest = load_manifest();
    std::set<int> unique_periods(requested_periods.begin(), requested_periods.end());
    for (const int period_number : unique_periods) {
        PeriodRecord period;
        try { period = load_period(period_number); } catch (...) { period.period = period_number; }
        const auto components = load_components_for_period(period_number);
        period.period = period_number;
        period.component_ids.clear();
        period.known_representative_count = 0;
        period.known_component_count_with_symmetry = 0;
        period.known_area = 0;
        period.known_area_error = 0;
        period.area_cutoff = area_cutoff;
        for (const auto& component : components) {
            period.component_ids.push_back(component.id);
            ++period.known_representative_count;
            period.known_component_count_with_symmetry += static_cast<std::size_t>(component.symmetry.multiplicity);
            period.known_area += component.geometry.area_estimate * component.symmetry.multiplicity;
            period.known_area_error += boost::multiprecision::abs(component.geometry.area_error)
                * component.symmetry.multiplicity;
        }
        std::sort(period.component_ids.begin(), period.component_ids.end());
        period.generated_from_catalogue_revision = manifest.catalogue_revision;
        save_period(period);
    }
    transaction.commit();
}

void Catalogue::rebuild_manifest_from_period_indexes(
    int exact_through_period,
    const CatalogueReal& minimum_area,
    const std::string& software_revision) const {
    ensure_layout();
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Manifest manifest = load_manifest();
    manifest.component_count_stored = 0;
    manifest.component_count_with_symmetry = 0;
    for (const int period_number : list_periods()) {
        try {
            const PeriodRecord period = load_period(period_number);
            manifest.component_count_stored += period.known_representative_count;
            manifest.component_count_with_symmetry += period.known_component_count_with_symmetry;
        } catch (...) {
        }
    }
    manifest.minimum_area = minimum_area;
    manifest.exact_through_period = exact_through_period;
    if (!software_revision.empty()) manifest.software_revision = software_revision;
    manifest.updated_at = utc_timestamp();
    save_manifest(manifest);
    transaction.commit();
}

std::size_t Catalogue::write_component_export(
    const fs::path& path,
    const ComponentExportOptions& options,
    const AreaScanProgressCallback& scan_progress,
    const AreaScanProgressCallback& write_progress) const {
    const auto components = query_components(options.query, scan_progress);
    Array rows;
    rows.reserve(components.size());
    const std::size_t write_steps = components.size() + 1;
    if (write_progress) write_progress(0, write_steps);
    for (std::size_t index = 0; index < components.size(); ++index) {
        const auto& component = components[index];
        Object row = as_object(component_to_json(component));
        const auto absolute = absolute_polygon(component);
        Array points;
        points.reserve(absolute.size());
        for (const auto& point : absolute) {
            points.push_back(complex_json(point, options.coordinate_digits));
        }
        row["absolutePolygon"] = std::move(points);
        rows.push_back(std::move(row));
        if (write_progress
            && ((index + 1) % 64 == 0
                || index + 1 == components.size())) {
            write_progress(index + 1, write_steps);
        }
    }
    const Manifest manifest = load_manifest();
    atomic_write(path, Object{
        {"format", options.format},
        {"complete", options.complete},
        {"catalogueRevision", integer_json(manifest.catalogue_revision)},
        {"components", std::move(rows)}});
    if (write_progress) write_progress(write_steps, write_steps);
    return components.size();
}

void Catalogue::write_skeleton_export(
    const fs::path& path,
    const ComponentQuery& query) const {
    const auto components = query_components(query);
    Array nodes;
    Array edges;
    for (const auto& component : components) {
        nodes.push_back(Object{
            {"id", component.id},
            {"period", std::to_string(component.period)},
            {"center", complex_json(component.center, component.numeric.working_precision_digits)},
            {"generation", component.hierarchy.generation
                ? Json(std::to_string(*component.hierarchy.generation)) : Json(nullptr)}});
        if (component.hierarchy.geometric_parent) {
            edges.push_back(Array{*component.hierarchy.geometric_parent, component.id});
        }
    }
    atomic_write(path, Object{
        {"format", "mandelbrot-component-skeleton-v1"},
        {"nodes", std::move(nodes)},
        {"edges", std::move(edges)}});
}

AreaScanStore Catalogue::area_scan_store(const std::string& run_name) const {
    return AreaScanStore(root_, run_name);
}

void Catalogue::verify_integrity() const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement integrity(database_->handle(), "PRAGMA integrity_check;");
    bool saw_row = false;
    while (true) {
        const int result = integrity.step();
        if (result == SQLITE_DONE) break;
        if (result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not run SQLite integrity check"));
        }
        saw_row = true;
        const std::string message = integrity.column_text(0);
        if (message != "ok") {
            throw std::runtime_error(
                "SQLite catalogue integrity check failed: " + message);
        }
    }
    if (!saw_row) {
        throw std::runtime_error(
            "SQLite catalogue integrity check returned no result");
    }

    Statement foreign_keys(database_->handle(), "PRAGMA foreign_key_check;");
    const int result = foreign_keys.step();
    if (result == SQLITE_ROW) {
        throw std::runtime_error(
            "SQLite catalogue contains a foreign-key violation in table "
            + foreign_keys.column_text(0));
    }
    if (result != SQLITE_DONE) {
        throw std::runtime_error(sqlite_error(
            database_->handle(), "Could not run foreign-key check"));
    }
}

AreaScanStore::AreaScanStore(fs::path catalogue_root, std::string run_name)
    : root_(std::move(catalogue_root)),
      exports_(root_ / "exports"),
      run_dir_(root_ / "runs" / "area_scan" / run_name),
      run_name_(std::move(run_name)),
      database_(std::make_shared<CatalogueDatabase>(
          root_ / "component_catalogue.sqlite")) {
    if (run_name_.empty()) {
        throw std::invalid_argument("Area-scan run name must not be empty");
    }
    fs::create_directories(exports_);
    fs::create_directories(run_dir_ / "root_checkpoints");
}

fs::path AreaScanStore::centers_path() const { return exports_ / "centers.csv"; }
fs::path AreaScanStore::measurements_path() const { return exports_ / "components.csv"; }
fs::path AreaScanStore::summary_path() const { return exports_ / "period_summary.csv"; }
fs::path AreaScanStore::root_checkpoint_path(int period) const {
    std::ostringstream name;
    name << "period_" << std::setw(2) << std::setfill('0') << period << ".chk";
    return run_dir_ / "root_checkpoints" / name.str();
}

std::map<int, std::vector<AreaScanCenterRecord>> AreaScanStore::load_centers(
    const AreaScanProgressCallback& progress
) const {
    const std::vector<std::string> headers{
        "period", "component_index", "expected_period_count", "center_re", "center_im",
        "center_residual", "detected_exact_period", "conjugate_index",
        "center_newton_iterations", "center_refinement_method", "center_refinement_dps"};
    std::map<int, std::vector<AreaScanCenterRecord>> result;
    auto decode = [&](const CsvObject& row) {
        try {
            AreaScanCenterRecord record;
            record.period = std::stoi(row.at("period"));
            record.component_index = std::stoi(row.at("component_index"));
            record.expected_period_count = std::stoi(row.at("expected_period_count"));
            record.center = {
                Catalogue::parse_decimal(row.at("center_re")),
                Catalogue::parse_decimal(row.at("center_im"))};
            record.center_residual = Catalogue::parse_decimal(row.at("center_residual"));
            record.detected_exact_period = std::stoi(row.at("detected_exact_period"));
            record.conjugate_index = std::stoi(row.at("conjugate_index"));
            if (const auto found = row.find("center_newton_iterations");
                found != row.end() && !found->second.empty()) {
                record.center_newton_iterations = std::stoi(found->second);
            }
            if (const auto found = row.find("center_refinement_method");
                found != row.end()) {
                record.center_refinement_method = found->second;
            }
            if (const auto found = row.find("center_refinement_dps");
                found != row.end() && !found->second.empty()) {
                record.center_refinement_dps = std::stoi(found->second);
            }
            result[record.period].push_back(std::move(record));
        } catch (...) {
        }
    };

    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run(
        *database_,
        "SELECT COUNT(*) FROM area_scan_centers WHERE run_name = ?1;",
        run_name_,
        "Could not count area-scan centers");
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_centers "
        "WHERE run_name = ?1 ORDER BY period, component_index;");
    statement.bind_text(1, run_name_);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not load area-scan centers"));
        }
        decode(csv_object_from_row(headers, statement.column_text(0)));
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) {
        progress(loaded, total);
    }
    for (auto& [_, records] : result) {
        std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
            return a.component_index < b.component_index;
        });
    }
    return result;
}

std::vector<AreaScanCenterRecord> AreaScanStore::load_centers(
    int period,
    const AreaScanProgressCallback& progress
) const {
    const std::vector<std::string> headers{
        "period", "component_index", "expected_period_count", "center_re", "center_im",
        "center_residual", "detected_exact_period", "conjugate_index",
        "center_newton_iterations", "center_refinement_method", "center_refinement_dps"};
    std::vector<AreaScanCenterRecord> result;
    auto decode = [&](const CsvObject& row) {
        try {
            AreaScanCenterRecord record;
            record.period = std::stoi(row.at("period"));
            record.component_index = std::stoi(row.at("component_index"));
            record.expected_period_count = std::stoi(row.at("expected_period_count"));
            record.center = {
                Catalogue::parse_decimal(row.at("center_re")),
                Catalogue::parse_decimal(row.at("center_im"))};
            record.center_residual = Catalogue::parse_decimal(row.at("center_residual"));
            record.detected_exact_period = std::stoi(row.at("detected_exact_period"));
            record.conjugate_index = std::stoi(row.at("conjugate_index"));
            if (const auto found = row.find("center_newton_iterations");
                found != row.end() && !found->second.empty()) {
                record.center_newton_iterations = std::stoi(found->second);
            }
            if (const auto found = row.find("center_refinement_method");
                found != row.end()) {
                record.center_refinement_method = found->second;
            }
            if (const auto found = row.find("center_refinement_dps");
                found != row.end() && !found->second.empty()) {
                record.center_refinement_dps = std::stoi(found->second);
            }
            if (record.period == period) result.push_back(std::move(record));
        } catch (...) {
        }
    };

    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run_period(
        *database_,
        "SELECT COUNT(*) FROM area_scan_centers "
        "WHERE run_name = ?1 AND period = ?2;",
        run_name_, period, "Could not count area-scan centers for period");
    result.reserve(total);
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_centers "
        "WHERE run_name = ?1 AND period = ?2 ORDER BY component_index;");
    statement.bind_text(1, run_name_);
    statement.bind_int(2, period);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not load area-scan centers for period"));
        }
        decode(csv_object_from_row(headers, statement.column_text(0)));
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) progress(loaded, total);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.component_index < b.component_index;
    });
    return result;
}

void AreaScanStore::save_centers(
    const std::map<int, std::vector<AreaScanCenterRecord>>& centers,
    const AreaScanProgressCallback& progress
) const {
    std::vector<const AreaScanCenterRecord*> records;
    for (const auto& [_, period_records] : centers) {
        for (const auto& record : period_records) records.push_back(&record);
    }
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_centers WHERE run_name = ?1;");
    remove.bind_text(1, run_name_);
    remove.expect_done("Could not replace area-scan centers");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_centers("
        "run_name, period, component_index, row_csv"
        ") VALUES(?1, ?2, ?3, ?4) "
        "ON CONFLICT(run_name, period, component_index) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, records.size());
    for (std::size_t row_index = 0; row_index < records.size(); ++row_index) {
        const auto& record = *records[row_index];
        const std::vector<std::string> values{
            std::to_string(record.period),
            std::to_string(record.component_index),
            std::to_string(record.expected_period_count),
            Catalogue::decimal_string(record.center.re),
            Catalogue::decimal_string(record.center.im),
            Catalogue::decimal_string(record.center_residual),
            std::to_string(record.detected_exact_period),
            std::to_string(record.conjugate_index),
            std::to_string(record.center_newton_iterations),
            record.center_refinement_method,
            std::to_string(record.center_refinement_dps)};
        insert.bind_text(1, run_name_);
        insert.bind_int(2, record.period);
        insert.bind_int(3, record.component_index);
        insert.bind_text(4, csv_row_text(values));
        insert.expect_done("Could not save area-scan center");
        insert.reset();
        if (progress
            && ((row_index + 1) % 1024 == 0
                || row_index + 1 == records.size())) {
            progress(row_index + 1, records.size());
        }
    }
    transaction.commit();
}

void AreaScanStore::save_centers(
    int period,
    const std::vector<AreaScanCenterRecord>& centers,
    const AreaScanProgressCallback& progress
) const {
    for (const auto& record : centers) {
        if (record.period != period) {
            throw std::invalid_argument(
                "Period-scoped center save received a row from another period");
        }
    }
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_centers WHERE run_name = ?1 AND period = ?2;");
    remove.bind_text(1, run_name_);
    remove.bind_int(2, period);
    remove.expect_done("Could not replace area-scan centers for period");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_centers("
        "run_name, period, component_index, row_csv"
        ") VALUES(?1, ?2, ?3, ?4) "
        "ON CONFLICT(run_name, period, component_index) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, centers.size());
    for (std::size_t index = 0; index < centers.size(); ++index) {
        const auto& record = centers[index];
        const std::vector<std::string> values{
            std::to_string(record.period),
            std::to_string(record.component_index),
            std::to_string(record.expected_period_count),
            Catalogue::decimal_string(record.center.re),
            Catalogue::decimal_string(record.center.im),
            Catalogue::decimal_string(record.center_residual),
            std::to_string(record.detected_exact_period),
            std::to_string(record.conjugate_index),
            std::to_string(record.center_newton_iterations),
            record.center_refinement_method,
            std::to_string(record.center_refinement_dps)};
        insert.bind_text(1, run_name_);
        insert.bind_int(2, period);
        insert.bind_int(3, record.component_index);
        insert.bind_text(4, csv_row_text(values));
        insert.expect_done("Could not save area-scan center for period");
        insert.reset();
        if (progress && ((index + 1) % 1024 == 0 || index + 1 == centers.size())) {
            progress(index + 1, centers.size());
        }
    }
    transaction.commit();
}

std::vector<AreaMeasurementRecord> AreaScanStore::load_measurements(
    const AreaScanProgressCallback& progress
) const {
    std::vector<AreaMeasurementRecord> result;
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run(
        *database_,
        "SELECT COUNT(*) FROM area_scan_measurements WHERE run_name = ?1;",
        run_name_,
        "Could not count area-scan measurements");
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_measurements "
        "WHERE run_name = ?1 "
        "ORDER BY period, component_index, rho_text;");
    statement.bind_text(1, run_name_);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(),
                "Could not load area-scan measurements"));
        }
        try {
            result.push_back(measurement_from_csv(csv_object_from_row(
                measurement_headers(), statement.column_text(0))));
        } catch (...) {
        }
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) {
        progress(loaded, total);
    }
    return result;
}

std::vector<AreaMeasurementRecord> AreaScanStore::load_measurements(
    int period,
    const AreaScanProgressCallback& progress
) const {
    std::vector<AreaMeasurementRecord> result;
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run_period(
        *database_,
        "SELECT COUNT(*) FROM area_scan_measurements "
        "WHERE run_name = ?1 AND period = ?2;",
        run_name_, period, "Could not count area-scan measurements for period");
    result.reserve(total);
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_measurements "
        "WHERE run_name = ?1 AND period = ?2 "
        "ORDER BY component_index, rho_text;");
    statement.bind_text(1, run_name_);
    statement.bind_int(2, period);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(),
                "Could not load area-scan measurements for period"));
        }
        try {
            result.push_back(measurement_from_csv(csv_object_from_row(
                measurement_headers(), statement.column_text(0))));
        } catch (...) {
        }
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) progress(loaded, total);
    return result;
}

std::vector<AreaMeasurementRecord> AreaScanStore::load_measurements(
    int period,
    const CatalogueReal& rho,
    const AreaScanProgressCallback& progress
) const {
    std::vector<AreaMeasurementRecord> result;
    const std::string rho_text = Catalogue::decimal_string(rho);
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run_period_rho(
        *database_,
        "SELECT COUNT(*) FROM area_scan_measurements "
        "WHERE run_name = ?1 AND period = ?2 AND rho_text = ?3;",
        run_name_, period, rho_text,
        "Could not count area-scan measurements for period and rho");
    result.reserve(total);
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_measurements "
        "WHERE run_name = ?1 AND period = ?2 AND rho_text = ?3 "
        "ORDER BY component_index;");
    statement.bind_text(1, run_name_);
    statement.bind_int(2, period);
    statement.bind_text(3, rho_text);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(),
                "Could not load area-scan measurements for period and rho"));
        }
        try {
            result.push_back(measurement_from_csv(csv_object_from_row(
                measurement_headers(), statement.column_text(0))));
        } catch (...) {
        }
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) progress(loaded, total);
    return result;
}

std::vector<AreaMeasurementRecord> AreaScanStore::load_measurements_from(
    const fs::path& path,
    const AreaScanProgressCallback& progress
) const {
    if (fs::weakly_canonical(path)
        == fs::weakly_canonical(measurements_path())) {
        return load_measurements(progress);
    }
    std::vector<AreaMeasurementRecord> result;
    for_each_csv_object(path, [&](const CsvObject& row) {
        try {
            result.push_back(measurement_from_csv(row));
        } catch (...) {
        }
    }, progress);
    return result;
}

void AreaScanStore::save_measurements(
    const std::vector<AreaMeasurementRecord>& measurements,
    const AreaScanProgressCallback& progress
) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_measurements WHERE run_name = ?1;");
    remove.bind_text(1, run_name_);
    remove.expect_done("Could not replace area-scan measurements");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_measurements("
        "run_name, period, component_index, rho_text, row_csv"
        ") VALUES(?1, ?2, ?3, ?4, ?5) "
        "ON CONFLICT(run_name, period, component_index, rho_text) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, measurements.size());
    for (std::size_t index = 0; index < measurements.size(); ++index) {
        const auto& record = measurements[index];
        const std::string rho = Catalogue::decimal_string(record.rho);
        insert.bind_text(1, run_name_);
        insert.bind_int(2, record.period);
        insert.bind_int(3, record.component_index);
        insert.bind_text(4, rho);
        insert.bind_text(5, csv_row_text(measurement_fields(record)));
        insert.expect_done("Could not save area-scan measurement");
        insert.reset();
        if (progress
            && ((index + 1) % 1024 == 0
                || index + 1 == measurements.size())) {
            progress(index + 1, measurements.size());
        }
    }
    transaction.commit();
}

void AreaScanStore::save_measurements(
    int period,
    const std::vector<AreaMeasurementRecord>& measurements,
    const AreaScanProgressCallback& progress
) const {
    for (const auto& record : measurements) {
        if (record.period != period) {
            throw std::invalid_argument(
                "Period-scoped measurement save received a row from another period");
        }
    }
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_measurements WHERE run_name = ?1 AND period = ?2;");
    remove.bind_text(1, run_name_);
    remove.bind_int(2, period);
    remove.expect_done("Could not replace area-scan measurements for period");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_measurements("
        "run_name, period, component_index, rho_text, row_csv"
        ") VALUES(?1, ?2, ?3, ?4, ?5) "
        "ON CONFLICT(run_name, period, component_index, rho_text) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, measurements.size());
    for (std::size_t index = 0; index < measurements.size(); ++index) {
        const auto& record = measurements[index];
        const std::string rho_text = Catalogue::decimal_string(record.rho);
        insert.bind_text(1, run_name_);
        insert.bind_int(2, period);
        insert.bind_int(3, record.component_index);
        insert.bind_text(4, rho_text);
        insert.bind_text(5, csv_row_text(measurement_fields(record)));
        insert.expect_done("Could not save area-scan measurement for period");
        insert.reset();
        if (progress
            && ((index + 1) % 1024 == 0 || index + 1 == measurements.size())) {
            progress(index + 1, measurements.size());
        }
    }
    transaction.commit();
}

void AreaScanStore::save_measurements_to(
    const fs::path& path,
    const std::vector<AreaMeasurementRecord>& measurements,
    const AreaScanProgressCallback& progress
) const {
    if (fs::weakly_canonical(path)
        == fs::weakly_canonical(measurements_path())) {
        save_measurements(measurements, progress);
        return;
    }
    write_csv_rows_stream(
        path, measurement_headers(), measurements.size(),
        [&](std::size_t row_index) {
            return measurement_fields(measurements[row_index]);
        },
        progress);
}

std::vector<AreaPeriodSummaryRecord> AreaScanStore::load_summaries(
    const AreaScanProgressCallback& progress
) const {
    const std::vector<std::string> headers{
        "period", "rho", "expected_components", "completed_components", "converged_components",
        "missing_or_unconverged_components", "period_complete", "min_area", "p10_area",
        "median_area", "mean_area", "p90_area", "max_area", "period_area", "cumulative_area",
        "cumulative_complete_through_period", "summed_error_estimate",
        "radial_increment_from_previous_rho"};
    std::vector<AreaPeriodSummaryRecord> result;
    auto decode = [&](const CsvObject& row) {
        try {
            AreaPeriodSummaryRecord r;
            r.period = std::stoi(row.at("period"));
            r.rho = Catalogue::parse_decimal(row.at("rho"));
            r.expected_components = std::stoi(row.at("expected_components"));
            r.completed_components = std::stoi(row.at("completed_components"));
            r.converged_components = std::stoi(row.at("converged_components"));
            r.missing_or_unconverged_components = std::stoi(row.at("missing_or_unconverged_components"));
            r.period_complete = parse_cache_bool(row.at("period_complete"));
            r.min_area = parse_optional_decimal(row.at("min_area"));
            r.p10_area = parse_optional_decimal(row.at("p10_area"));
            r.median_area = parse_optional_decimal(row.at("median_area"));
            r.mean_area = parse_optional_decimal(row.at("mean_area"));
            r.p90_area = parse_optional_decimal(row.at("p90_area"));
            r.max_area = parse_optional_decimal(row.at("max_area"));
            r.period_area = Catalogue::parse_decimal(row.at("period_area"));
            r.cumulative_area = Catalogue::parse_decimal(row.at("cumulative_area"));
            r.cumulative_complete_through_period = parse_cache_bool(row.at("cumulative_complete_through_period"));
            r.summed_error_estimate = Catalogue::parse_decimal(row.at("summed_error_estimate"));
            r.radial_increment_from_previous_rho = parse_optional_decimal(row.at("radial_increment_from_previous_rho"));
            result.push_back(std::move(r));
        } catch (...) {
        }
    };
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    const std::size_t total = count_rows_for_run(
        *database_,
        "SELECT COUNT(*) FROM area_scan_summaries WHERE run_name = ?1;",
        run_name_,
        "Could not count area-scan summaries");
    Statement statement(
        database_->handle(),
        "SELECT row_csv FROM area_scan_summaries "
        "WHERE run_name = ?1 ORDER BY period, rho_text;");
    statement.bind_text(1, run_name_);
    std::size_t loaded = 0;
    if (progress) progress(0, total);
    while (true) {
        const int step_result = statement.step();
        if (step_result == SQLITE_DONE) break;
        if (step_result != SQLITE_ROW) {
            throw std::runtime_error(sqlite_error(
                database_->handle(), "Could not load area-scan summaries"));
        }
        decode(csv_object_from_row(headers, statement.column_text(0)));
        ++loaded;
        if (progress && loaded % 1024 == 0) progress(loaded, total);
    }
    if (progress && (loaded == 0 || loaded % 1024 != 0)) {
        progress(loaded, total);
    }
    return result;
}

bool AreaScanStore::has_summaries() const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(),
        "SELECT 1 FROM area_scan_summaries "
        "WHERE run_name = ?1 LIMIT 1;");
    statement.bind_text(1, run_name_);
    const int result = statement.step();
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    throw std::runtime_error(sqlite_error(
        database_->handle(),
        "Could not test area-scan summary existence"));
}

bool AreaScanStore::has_summaries(int period) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Statement statement(
        database_->handle(),
        "SELECT 1 FROM area_scan_summaries "
        "WHERE run_name = ?1 AND period = ?2 LIMIT 1;");
    statement.bind_text(1, run_name_);
    statement.bind_int(2, period);
    const int result = statement.step();
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    throw std::runtime_error(sqlite_error(
        database_->handle(),
        "Could not test area-scan summary existence for period"));
}

void AreaScanStore::save_summaries(
    const std::vector<AreaPeriodSummaryRecord>& summaries,
    const AreaScanProgressCallback& progress
) const {
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_summaries WHERE run_name = ?1;");
    remove.bind_text(1, run_name_);
    remove.expect_done("Could not replace area-scan summaries");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_summaries("
        "run_name, period, rho_text, row_csv"
        ") VALUES(?1, ?2, ?3, ?4) "
        "ON CONFLICT(run_name, period, rho_text) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, summaries.size());
    for (std::size_t row_index = 0; row_index < summaries.size(); ++row_index) {
        const auto& r = summaries[row_index];
        const std::string rho = Catalogue::decimal_string(r.rho);
        const std::vector<std::string> fields{
            std::to_string(r.period),
            rho,
            std::to_string(r.expected_components),
            std::to_string(r.completed_components),
            std::to_string(r.converged_components),
            std::to_string(r.missing_or_unconverged_components),
            r.period_complete ? "True" : "False",
            optional_decimal_string(r.min_area),
            optional_decimal_string(r.p10_area),
            optional_decimal_string(r.median_area),
            optional_decimal_string(r.mean_area),
            optional_decimal_string(r.p90_area),
            optional_decimal_string(r.max_area),
            Catalogue::decimal_string(r.period_area),
            Catalogue::decimal_string(r.cumulative_area),
            r.cumulative_complete_through_period ? "True" : "False",
            Catalogue::decimal_string(r.summed_error_estimate),
            optional_decimal_string(r.radial_increment_from_previous_rho)};
        insert.bind_text(1, run_name_);
        insert.bind_int(2, r.period);
        insert.bind_text(3, rho);
        insert.bind_text(4, csv_row_text(fields));
        insert.expect_done("Could not save area-scan summary");
        insert.reset();
        if (progress
            && ((row_index + 1) % 1024 == 0
                || row_index + 1 == summaries.size())) {
            progress(row_index + 1, summaries.size());
        }
    }
    transaction.commit();
}

void AreaScanStore::save_summaries(
    int period,
    const std::vector<AreaPeriodSummaryRecord>& summaries,
    const AreaScanProgressCallback& progress
) const {
    for (const auto& row : summaries) {
        if (row.period != period) {
            throw std::invalid_argument(
                "Period-scoped summary save received a row from another period");
        }
    }
    std::lock_guard<std::recursive_mutex> lock(database_->mutex);
    Transaction transaction(*database_);
    Statement remove(
        database_->handle(),
        "DELETE FROM area_scan_summaries WHERE run_name = ?1 AND period = ?2;");
    remove.bind_text(1, run_name_);
    remove.bind_int(2, period);
    remove.expect_done("Could not replace area-scan summaries for period");

    Statement insert(
        database_->handle(),
        "INSERT INTO area_scan_summaries("
        "run_name, period, rho_text, row_csv"
        ") VALUES(?1, ?2, ?3, ?4) "
        "ON CONFLICT(run_name, period, rho_text) "
        "DO UPDATE SET row_csv = excluded.row_csv;");
    if (progress) progress(0, summaries.size());
    for (std::size_t index = 0; index < summaries.size(); ++index) {
        const auto& row = summaries[index];
        const std::string rho_text = Catalogue::decimal_string(row.rho);
        const std::vector<std::string> fields{
            std::to_string(row.period),
            rho_text,
            std::to_string(row.expected_components),
            std::to_string(row.completed_components),
            std::to_string(row.converged_components),
            std::to_string(row.missing_or_unconverged_components),
            row.period_complete ? "True" : "False",
            optional_decimal_string(row.min_area),
            optional_decimal_string(row.p10_area),
            optional_decimal_string(row.median_area),
            optional_decimal_string(row.mean_area),
            optional_decimal_string(row.p90_area),
            optional_decimal_string(row.max_area),
            Catalogue::decimal_string(row.period_area),
            Catalogue::decimal_string(row.cumulative_area),
            row.cumulative_complete_through_period ? "True" : "False",
            Catalogue::decimal_string(row.summed_error_estimate),
            optional_decimal_string(row.radial_increment_from_previous_rho)};
        insert.bind_text(1, run_name_);
        insert.bind_int(2, period);
        insert.bind_text(3, rho_text);
        insert.bind_text(4, csv_row_text(fields));
        insert.expect_done("Could not save area-scan summary for period");
        insert.reset();
        if (progress
            && ((index + 1) % 1024 == 0 || index + 1 == summaries.size())) {
            progress(index + 1, summaries.size());
        }
    }
    transaction.commit();
}

CatalogueReal Catalogue::parse_decimal(const std::string& value) {
    try {
        CatalogueReal result(value);
        if (!boost::multiprecision::isfinite(result)) {
            throw std::invalid_argument("non-finite decimal");
        }
        return result;
    } catch (const std::exception& error) {
        throw std::invalid_argument("Invalid catalogue decimal '" + value + "': " + error.what());
    }
}

std::string Catalogue::decimal_string(const CatalogueReal& value, int digits) {
    if (!boost::multiprecision::isfinite(value)) {
        throw std::invalid_argument("Cannot serialize non-finite catalogue decimal");
    }
    if (value == 0) return "0";

    // Boost's scientific precision is the number of digits *after* the
    // decimal point, while this API uses total significant digits.
    const int significant_digits = digits > 0
        ? digits
        : std::numeric_limits<CatalogueReal>::max_digits10;
    const int precision = std::max(0, significant_digits - 1);
    std::string raw = value.str(precision, std::ios_base::scientific);

    bool negative = false;
    if (!raw.empty() && (raw.front() == '-' || raw.front() == '+')) {
        negative = raw.front() == '-';
        raw.erase(raw.begin());
    }
    const std::size_t exponent_position = raw.find_first_of("eE");
    if (exponent_position == std::string::npos) return (negative ? "-" : "") + raw;

    std::string mantissa = raw.substr(0, exponent_position);
    int exponent = std::stoi(raw.substr(exponent_position + 1));
    mantissa.erase(std::remove(mantissa.begin(), mantissa.end(), '.'), mantissa.end());
    while (mantissa.size() > 1 && mantissa.back() == '0') mantissa.pop_back();
    if (mantissa.find_first_not_of('0') == std::string::npos) return "0";

    const std::string sign = negative ? "-" : "";
    std::string scientific = sign;
    scientific.push_back(mantissa.front());
    if (mantissa.size() > 1) {
        scientific.push_back('.');
        scientific.append(mantissa.begin() + 1, mantissa.end());
    }
    if (exponent != 0) scientific += "e" + std::to_string(exponent);

    const long long point = 1LL + static_cast<long long>(exponent);
    std::string plain = sign;
    if (point <= 0) {
        plain += "0.";
        plain.append(static_cast<std::size_t>(-point), '0');
        plain += mantissa;
    } else if (point >= static_cast<long long>(mantissa.size())) {
        plain += mantissa;
        plain.append(static_cast<std::size_t>(point - static_cast<long long>(mantissa.size())), '0');
    } else {
        plain += mantissa.substr(0, static_cast<std::size_t>(point));
        plain.push_back('.');
        plain += mantissa.substr(static_cast<std::size_t>(point));
    }
    return plain.size() <= scientific.size() ? plain : scientific;
}

} // namespace mandelbrot::catalogue
