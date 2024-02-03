#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // Open the input files
    ifstream win_prob_file("log_file_win_prob.txt");
    ifstream policy_net_loss_file("log_file_policy_net_loss.txt");
    ifstream next_scene_net_loss_file("log_file_next_scene_net_loss.txt");

    // Open the output file
    ofstream output_file("output.txt");

    // Write the header line to the output file
    output_file << "epoch Old_win_probablilty total_plays epsilon Policy_net_Loss scene_predictiable_net_Loss" << endl;

    // Parse the data from the input files and write to the output file
    string win_prob_line, policy_net_loss_line, next_scene_net_loss_line;
    while (getline(win_prob_file, win_prob_line) && getline(policy_net_loss_file, policy_net_loss_line) && getline(next_scene_net_loss_file, next_scene_net_loss_line)) {
        // Parse the epoch number
        size_t epoch_pos = win_prob_line.find("epoch = ");
        int epoch = stoi(win_prob_line.substr(epoch_pos + 8));

        // Parse the Old win probability and total plays from the win prob file
        size_t old_win_prob_pos = win_prob_line.find("Old win probablilty = ");
        float old_win_prob = stof(win_prob_line.substr(old_win_prob_pos + 22));
        size_t total_plays_pos = win_prob_line.find("total plays = ");
        int total_plays = stoi(win_prob_line.substr(total_plays_pos + 14));

        // Parse the epsilon from the policy net loss file
        size_t epsilon_pos = policy_net_loss_line.find("epsilon = ");
        float epsilon = stof(policy_net_loss_line.substr(epsilon_pos + 10));

        // Parse the policy net loss from the policy net loss file
        size_t policy_net_loss_pos = policy_net_loss_line.find("Policy net Loss = ");
        float policy_net_loss = stof(policy_net_loss_line.substr(policy_net_loss_pos + 18));

        // Parse the scene predictiable net loss from the next scene net loss file
        size_t scene_predictiable_net_loss_pos = next_scene_net_loss_line.find("scene predictiable net Loss = ");
        float scene_predictiable_net_loss = stof(next_scene_net_loss_line.substr(scene_predictiable_net_loss_pos + 31));

        // Write the parsed data to the output file
        output_file << epoch << " " << old_win_prob << " " << total_plays << " " << epsilon << " " << policy_net_loss << " " << scene_predictiable_net_loss << endl;
    }

    // Close the input and output files
    win_prob_file.close();
    policy_net_loss_file.close();
    next_scene_net_loss_file.close();
    output_file.close();

    cout << "Output written to output.txt" << endl;

    return 0;
}
