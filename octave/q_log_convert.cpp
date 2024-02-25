#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // Open the input files
    ifstream win_prob_file("log_file_win_prob.txt");
    ifstream quality_net_loss_file("log_file_quality_net_loss.txt");
    ifstream next_scene_net_loss_file("log_file_next_scene_net_loss.txt");
    string win_prob_line, quality_net_loss_line, next_scene_net_loss_line;

    // Open the output file
    ofstream log_file_win_prob_stripped_file("log_file_win_prob_stripped.txt");
    ofstream quality_net_loss_stripped_file("quality_net_loss_stripped.txt");
    ofstream next_scene_net_loss_stripped_file("next_scene_net_loss_stripped.txt");
 
    // Process win_prob_file
    while (getline(win_prob_file, win_prob_line)) {
        if (win_prob_line.find("epoch") == 0) {
            log_file_win_prob_stripped_file << win_prob_line << endl;
        }
    }
    // Close the file log_file_win_prob_stripped_file
    log_file_win_prob_stripped_file.close();

    // Process quality_net_loss_file
    while (getline(quality_net_loss_file, quality_net_loss_line)) {
        if (quality_net_loss_line.find("epoch") == 0) {
            quality_net_loss_stripped_file << quality_net_loss_line << endl;
        }
    }
    // Close the file quality_net_loss_stripped_file
    quality_net_loss_stripped_file.close();

    // Process next_scene_net_loss_file
    while (getline(next_scene_net_loss_file, next_scene_net_loss_line)) {
        if (next_scene_net_loss_line.find("epoch") == 0) {
            next_scene_net_loss_stripped_file << next_scene_net_loss_line << endl;
        }
    }
    // Close the file next_scene_net_loss_stripped_file
    next_scene_net_loss_stripped_file.close();


    // Close the input and output files
    quality_net_loss_file.close();
    next_scene_net_loss_file.close();
    quality_net_loss_file.close();


    // Open the output file
    ofstream output_file_win("output_win.txt");
    ofstream output_file_policy("output_policy.txt");
    ofstream output_file_next_scene("output_next_scene.txt");

    // Open the input files
    ifstream win_prob_stripped_file_inp("log_file_win_prob_stripped.txt");
    ifstream quality_net_loss_stripped_file_inp("quality_net_loss_stripped.txt");
    ifstream next_scene_net_loss_stripped_file_inp("next_scene_net_loss_stripped.txt");
    
    // Write the header line to the output file
    output_file_win << "epoch Now_win_probablilty Old_win_probablilty total_plays" << endl;
    output_file_policy << "epoch epsilon quality_net_Loss" << endl;
    output_file_next_scene << "epoch scene_predictiable_net_Loss" << endl;    
    while (getline(win_prob_stripped_file_inp, win_prob_line))
    {
        // Parse the epoch number
        size_t epoch_pos = win_prob_line.find("epoch = ");
        cout << epoch_pos << endl;
       // int epoch_pos_int = win_prob_line.find("Old win probablilty = ");
        int epoch = 0;
        if (epoch_pos != string::npos)
        {
            epoch = stoi(win_prob_line.substr(epoch_pos + 8));
            cout << "The integer value after \"epoch = \" is: " << epoch << endl;
        }
        //Win probaility Now = 
        // Parse the Now win probability and total plays from the win prob file
        size_t now_win_prob_pos = win_prob_line.find("Win probaility Now = ");
        float now_win_prob = 0.0f;
        if (now_win_prob_pos != string::npos)
        {
            now_win_prob = stof(win_prob_line.substr(now_win_prob_pos + 21));
        }
        // Parse the Old win probability and total plays from the win prob file
        size_t old_win_prob_pos = win_prob_line.find("Old win probablilty = ");
        float old_win_prob = 0.0f;
        if (old_win_prob_pos != string::npos)
        {
            old_win_prob = stof(win_prob_line.substr(old_win_prob_pos + 22));
        }
        size_t total_plays_pos = win_prob_line.find("total plays = ");
        int total_plays = 0;
        if (total_plays_pos != string::npos)
        {
            total_plays = stoi(win_prob_line.substr(total_plays_pos + 14));
        }
        // Write the parsed data to the output file
        output_file_win << epoch << " " << now_win_prob << " " << old_win_prob << " " << total_plays << endl;
    }


    while (getline(quality_net_loss_stripped_file_inp, quality_net_loss_line)) 
    {
        // Parse the epoch number
        size_t epoch_pos = quality_net_loss_line.find("epoch = ");
        cout << epoch_pos << endl;
       // int epoch_pos_int = win_prob_line.find("Old win probablilty = ");
        int epoch = 0;
        if (epoch_pos != string::npos)
        {
            epoch = stoi(quality_net_loss_line.substr(epoch_pos + 8));
            cout << "The integer value after \"epoch = \" is: " << epoch << endl;
        }

        // Parse the epsilon from the policy net loss file
        size_t epsilon_pos = quality_net_loss_line.find("epsilon = ");
        float epsilon = 0.0f;
        if (epsilon_pos != string::npos)
        {
            epsilon = stof(quality_net_loss_line.substr(epsilon_pos + 10));
        }       
        // Parse the policy net loss from the policy net loss file
        size_t quality_net_loss_pos = quality_net_loss_line.find("Policy net Loss = ");
        float quality_net_loss = 0.0f;
        if (quality_net_loss_pos != string::npos)
        {
            quality_net_loss = stof(quality_net_loss_line.substr(quality_net_loss_pos + 18));
        }       
        // Write the parsed data to the output file
        output_file_policy << epoch << " " << epsilon << " " << quality_net_loss <<  endl;
    }

    while (getline(next_scene_net_loss_stripped_file_inp, next_scene_net_loss_line)) 
    {
        // Parse the epoch number
        size_t epoch_pos = next_scene_net_loss_line.find("epoch = ");
        cout << epoch_pos << endl;
       // int epoch_pos_int = win_prob_line.find("Old win probablilty = ");
        int epoch = 0;
        if (epoch_pos != string::npos)
        {
            epoch = stoi(next_scene_net_loss_line.substr(epoch_pos + 8));
            cout << "The integer value after \"epoch = \" is: " << epoch << endl;
        }
        
        // Parse the scene predictiable net loss from the next scene net loss file
        size_t scene_predictiable_net_loss_pos = next_scene_net_loss_line.find("scene predictiable net Loss = ");
        float scene_predictiable_net_loss = 0.0f;
        if (scene_predictiable_net_loss_pos != string::npos)
        {
            scene_predictiable_net_loss = stof(next_scene_net_loss_line.substr(scene_predictiable_net_loss_pos + 30));
        }       

        // Write the parsed data to the output file
        output_file_next_scene << epoch << " " << scene_predictiable_net_loss << endl;
    }


    // Close the input and output files
    win_prob_stripped_file_inp.close();
    quality_net_loss_stripped_file_inp.close();
    next_scene_net_loss_stripped_file_inp.close();
    output_file_win.close();
    output_file_policy.close();
    output_file_next_scene.close();

    return 0;

}