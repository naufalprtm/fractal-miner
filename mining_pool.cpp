#include <iostream>
#include <string>
#include <curl/curl.h>
#include "mining_pool.h"

// Write callback function to handle responses
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::string* str = static_cast<std::string*>(userp);
    size_t total_size = size * nmemb;
    str->append(static_cast<const char*>(contents), total_size);
    return total_size;
}

void connectToPool(const std::string& url, const std::string& user, const std::string& pass) {
    CURL *curl;
    CURLcode res;
    long response_code;
    std::string read_buffer;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer); // Set write data to capture response
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        // Subscribe to the mining pool
        std::string stratum_request = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[]}\n";
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, stratum_request.c_str());

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cout << "HTTP Response Code: " << response_code << std::endl;
            std::cout << "Response from pool: " << read_buffer << std::endl; // Print the response
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}
