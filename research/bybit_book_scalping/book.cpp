// Bybit's compact JSONL orderbook archive has string-valued price/size pairs.
// Rebuild every delta before sampling the first valid quote per five-second bucket.
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
using Book = std::map<double, double>;

void require(bool valid) {
    if (!valid) throw std::runtime_error("Unexpected Bybit orderbook data");
}

size_t field(std::string_view line, std::string_view key) {
    size_t start = line.find(key);
    require(start != line.npos);
    return start + key.size();
}

void levels(std::string_view line, std::string_view key, Book& book) {
    size_t pos = field(line, key);
    while (line.at(pos) != ']') {
        require(line.substr(pos, 2) == "[\"");
        const char* end = line.data() + line.size();
        double price, size;
        auto p = std::from_chars(line.data() + pos + 2, end, price);
        require(p.ec == std::errc{} && std::string_view(p.ptr, 3) == "\",\"");
        auto q = std::from_chars(p.ptr + 3, end, size);
        require(q.ec == std::errc{} && std::string_view(q.ptr, 2) == "\"]");
        require(price > 0 && size >= 0);
        if (size == 0) book.erase(price);
        else book[price] = size;
        pos = q.ptr - line.data() + 2;
        if (line.at(pos) == ',') ++pos;
    }
}

template <class Iterator>
double depth(Iterator it, Iterator end) {
    double quantity = 0;
    for (int count = 0; count < 5 && it != end; ++it, ++count)
        quantity += it->second;
    return quantity;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cout << std::setprecision(12);
    Book bids, asks;
    long long bucket = -1, previous = -1;
    bool ready = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        bool snapshot = line.find("\"type\":\"snapshot\"") != line.npos;
        auto update = std::stoll(line.substr(field(line, "\"u\":")));
        if (snapshot) {
            bids.clear(); asks.clear(); ready = true;
        } else if (update != previous + 1) {
            ready = false; // resume from a complete snapshot
        }
        previous = update;
        if (!ready) continue;
        levels(line, "\"b\":[", bids);
        levels(line, "\"a\":[", asks);
        auto ts = std::stoll(line.substr(field(line, "\"ts\":")));
        if (ts / 5000 == bucket || bids.empty() || asks.empty()) continue;
        auto bid = bids.rbegin();
        auto ask = asks.begin();
        require(bid->first < ask->first);
        bucket = ts / 5000;
        std::cout << ts << ',' << bid->first << ',' << ask->first << ','
                  << bid->second << ',' << ask->second << ','
                  << depth(bids.rbegin(), bids.rend()) << ','
                  << depth(asks.begin(), asks.end()) << '\n';
    }
}
