#include <iostream>
#include <string>
#include <curl/curl.h>
#include <cstring>      // For memset
#include <netdb.h>     // For addrinfo and getaddrinfo
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "mining_rpc.h"

// Write callback function to handle responses
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::string* str = static_cast<std::string*>(userp);
    size_t total_size = size * nmemb;
    str->append(static_cast<const char*>(contents), total_size);
    return total_size;
}

bool checkConnection(const std::string& url) {
    struct addrinfo hints, *res;
    int sock;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET; // IPv4
    hints.ai_socktype = SOCK_STREAM;

    // Resolve the hostname (update to use the actual URL passed)
    if (getaddrinfo(url.c_str(), "5002", &hints, &res) != 0) { // Using port 5002 for RPC
        std::cerr << "[ERROR] Could not resolve hostname" << std::endl;
        return false;
    }

    // Create a socket
    sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock == -1) {
        std::cerr << "[ERROR] Could not create socket" << std::endl;
        freeaddrinfo(res);
        return false;
    }

    // Try to connect
    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0) {
        std::cerr << "[ERROR] Connection failed" << std::endl;
        close(sock);
        freeaddrinfo(res);
        return false;
    }

    close(sock);
    freeaddrinfo(res);
    std::cout << "[INFO] Connection to " << url << " successful." << std::endl;
    return true;
}

void connectToRPC(const std::string& url, const std::string& user, const std::string& pass) {
    std::cout << "[INFO] Checking connection to the mining RPC..." << std::endl;

    // Check if the connection to the RPC is successful
    if (!checkConnection(url)) {
        std::cerr << "[ERROR] Failed to connect to the mining RPC." << std::endl;
        return;
    }

    CURL* curl;
    CURLcode res;
    long response_code;
    std::string read_buffer;

    std::cout << "[INFO] Initializing cURL..." << std::endl;
    curl = curl_easy_init();
    if (curl) {
        std::cout << "[INFO] Setting cURL options..." << std::endl;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer); // Set write data to capture response
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

        // Subscribe to the mining RPC
        std::string rpc_request = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[]}\n";
        std::cout << "[INFO] Sending RPC request: " << rpc_request << std::endl;
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, rpc_request.c_str());

        std::cout << "[INFO] Performing cURL request..." << std::endl;
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cout << "[INFO] HTTP Response Code: " << response_code << std::endl;
            std::cout << "[INFO] Response from RPC: " << read_buffer << std::endl; // Print the response
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    } else {
        std::cerr << "[ERROR] Failed to initialize cURL." << std::endl;
    }
}
