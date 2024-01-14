
#include <iostream>
#include <stdio.h>
#include "fc_m_resnet.hpp"
#include <vector>
#include <time.h>
using namespace std;
#include <cstdlib>
#include <ctime>
#include <math.h>   // exp
#include <stdlib.h> // exit(0);
#include <cstdlib>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)
#include <iomanip>               // for std::setprecision

#include "pinball_game.hpp"
#define MOVE_DOWN 0
#define MOVE_UP 1
#define MOVE_STOP 2



vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{
    int skip_scene_predicotr_only_for_benchmarking = 0;// set this to 1 to benchmaring with only use policy network like orderanry vanilla on policy reinforcemnat learning instead of dubble network with scenen predictor
    char answer;
    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer
    cout << "Scene prdictable reinforcement learning game" << endl;

    cout << "On going coding TODO. Not finnish...... " << endl;

    int total_plays = 0;
    //======================================================
    pinball_game gameObj1;      /// Instaniate the pinball game
    gameObj1.init_game();       /// Initialize the pinball game with serten parametrers
    gameObj1.slow_motion = 0;   /// 0=full speed game. 1= slow down
    gameObj1.replay_times = 0;  /// If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 0; /// 0= only a ball. 1= ball give awards. square gives punish
    gameObj1.use_image_diff = 0;
    gameObj1.high_precition_mode = 0; /// This will make adjustable rewards highest at center of the pad.
    gameObj1.use_dice_action = 0;
    gameObj1.drop_out_percent = 0;
    gameObj1.Not_dropout = 1;
    gameObj1.flip_reward_sign = 0;
    gameObj1.print_out_nodes = 0;
    gameObj1.enable_ball_swan = 0;
    gameObj1.use_character = 0;
    gameObj1.enabel_3_state = 1; // Input Action from Agent. move_up: 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1

    // Set up a OpenCV mat
    const int pixel_height = 35; /// The input data pixel height, note game_Width = 220
    const int pixel_width = 35;  /// The input data pixel width, note game_Height = 200
    Size image_size_reduced(pixel_height, pixel_width); // the dst image size,e.g.50x50
    Mat resized_grapics, game_video_full_size;
    const int nr_frames_strobed = 4; // 4 Images in serie to make neural network to see movments

    typedef struct
    {
        vector<double> video_frame; // vector of doubles to store the video frame
        int selected_action;        // integer to store the selected action
        int dice_used;              // integer to store if the dice was used = 1 or not used = 0
        int terminal_state;     // 1 = we are in terminal state. 0 normal state
        double rewards_Q;      // store the rewards at real game. Then recalculate this before training with a decay factor
    } replay_data_struct;

    replay_data_struct replay_data;
    replay_data.video_frame.resize(pixel_height * pixel_width);

    vector<replay_data_struct> replay_1_episode_data_buffer;
    for (int i = 0; i < gameObj1.nr_of_frames; ++i)
    {
        replay_data_struct replay_data_item;
        replay_data_item.video_frame.resize(pixel_height * pixel_width);
        replay_1_episode_data_buffer.push_back(replay_data_item);
    }


    //=========== Neural Network size settings ==============
    fc_m_resnet next_scene_fc_net;
    fc_m_resnet policy_fc_net;
    int save_cnt = 0;

    string next_scene_net_filename;
    next_scene_net_filename = "next_scene_net_weights.dat";

    const int all_clip_der = 0;

    next_scene_fc_net.get_version();
    next_scene_fc_net.block_type = 2;
    next_scene_fc_net.use_softmax = 0;                         // 0= Not softmax for DQN reinforcement learning
    next_scene_fc_net.activation_function_mode = 2;            // ReLU for all fully connected activation functions except output last layer
    next_scene_fc_net.force_last_activation_function_mode = 0; // 1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
    next_scene_fc_net.use_skip_connect_mode = 0;               // 1 for residual network architetcture
    next_scene_fc_net.use_dropouts = 0;
    next_scene_fc_net.dropout_proportion = 0.0;
    next_scene_fc_net.clip_deriv = all_clip_der;

    int nr_of_actions = 3;

    const int next_scene_hid_layers = 3;
    const int next_scene_hid_nodes_L1 = 200;
    const int next_scene_hid_nodes_L2 = 75;
    const int next_scene_hid_nodes_L3 = 200;
    const int next_scene_out_nodes = pixel_height * pixel_width; // 
    const int next_scene_inp_nodes = pixel_height * pixel_width * nr_frames_strobed + nr_of_actions;
    // replay_grapics_buffert.create(replay_row_size, replay_col_size, CV_32FC1);
    Mat mat_input_weights_next_sc, mat_input_strobe_frames, mat_next_scene_all_actions;
    mat_input_weights_next_sc.create(pixel_height * nr_frames_strobed * next_scene_hid_nodes_L1, pixel_width, CV_32FC1);
    mat_input_strobe_frames.create(pixel_height * nr_frames_strobed, pixel_width, CV_32FC1);
    mat_next_scene_all_actions.create(pixel_height * nr_of_actions, pixel_width, CV_32FC1);
    

    cout << "next_scene_inp_nodes = " << next_scene_inp_nodes << endl;
    
    for (int i = 0; i < next_scene_inp_nodes; i++)
    {
        next_scene_fc_net.input_layer.push_back(0.0);
        next_scene_fc_net.i_layer_delta.push_back(0.0);
    }

    for (int i = 0; i < next_scene_out_nodes; i++)
    {
        next_scene_fc_net.output_layer.push_back(0.0);
        next_scene_fc_net.target_layer.push_back(0.0);
    }
    next_scene_fc_net.set_nr_of_hidden_layers(next_scene_hid_layers);
    next_scene_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L1);
    next_scene_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L2);
    next_scene_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L3);
    //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(end_hid_layers)
    //============ Neural Network Size setup is finnish ! ==================

    //=== Now setup the hyper parameters of the Neural Network ====
    double reward_gain = 10.0;
    const double learning_rate_fc = 0.001;
    double learning_rate_end = learning_rate_fc;
    next_scene_fc_net.learning_rate = learning_rate_end;
    next_scene_fc_net.momentum = 0.95; //
    double init_random_weight_propotion = 0.25;
    const double warm_up_epsilon_default = 0.85;
    double warm_up_epsilon = warm_up_epsilon_default;
    const double warm_up_eps_derating = 0.15;
    int warm_up_eps_nr = 3;
    int warm_up_eps_cnt = 0;
    const double start_epsilon = 0.50;
    const double stop_min_epsilon = 0.05;
    const double derating_epsilon = 0.005;
    double epsilon = start_epsilon; // Exploring vs exploiting parameter weight if dice above this threshold chouse random action. If dice below this threshold select strongest outoput action node
    if (warm_up_eps_nr > 0)
    {
        epsilon = warm_up_epsilon;
    }
     cout << "Do you want to set a manual set epsilon value from start = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        cout << "Set a warm_up_epsilon value between " << warm_up_epsilon << " to stop_min_epsilon = " << stop_min_epsilon << endl;
        cin >> epsilon;
        if (epsilon > warm_up_epsilon_default)
        {
            epsilon = warm_up_epsilon_default;
        }
        if (epsilon < stop_min_epsilon)
        {
            epsilon = stop_min_epsilon;
        }
        cout << " epsilon is now set to = " << epsilon << endl;
        warm_up_eps_nr = 0;
    }

    cout << " epsilon = " << epsilon << endl;
    double gamma = 0.85f;
    const int g_replay_size = 1000; // Should be 10000 or more
    const int save_after_nr = 1;
    // statistics report
    const int max_w_p_nr = 1000;
    int win_p_cnt = 0;
    int win_counter = 0;
    double last_win_probability = 0.5;
    double now_win_probability = last_win_probability;


    vector<vector<replay_data_struct>> replay_buffer;
    for(int i=0;i<g_replay_size;i++)
    {
        replay_buffer.push_back(replay_1_episode_data_buffer);
    }

    cout << " gameObj1.nr_of_frames = " << gameObj1.nr_of_frames << endl;

    //==== Hyper parameter settings End ===========================


    cout << "Do you want to load kernel weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        next_scene_fc_net.load_weights(next_scene_net_filename);
    }
    else
    {
        next_scene_fc_net.randomize_weights(init_random_weight_propotion);
    }
    cout << "gameObj1.gameObj1.game_Height " << gameObj1.game_Height << endl;
    cout << "gameObj1.gameObj1.game_Width " << gameObj1.game_Width << endl;

    // Start onlu one game now, only for get out size of grapichs to prepare memory
    gameObj1.replay_episode = 0;
    gameObj1.start_episode();

    game_video_full_size = gameObj1.gameGrapics.clone();
    resize(game_video_full_size, resized_grapics, image_size_reduced);
    imshow("resized_grapics", resized_grapics); ///  resize(src, dst, size);


    const int max_nr_epochs = 1000000;
    for (int epoch = 0; epoch < max_nr_epochs; epoch++)
    {
        cout << "******** Epoch number = " << epoch << " **********" << endl;
        cout << "epsilon = " << epsilon << endl;
        for (int g_replay_cnt = 0; g_replay_cnt < g_replay_size; g_replay_cnt++)
        {
            gameObj1.start_episode();
            for (int frame_g = 0; frame_g < gameObj1.nr_of_frames; frame_g++) // Loop throue each of the 100 frames
            {
                gameObj1.frame = frame_g;
                gameObj1.run_episode();
                game_video_full_size = gameObj1.gameGrapics.clone();
                resize(game_video_full_size, resized_grapics, image_size_reduced);
                imshow("resized_grapics", resized_grapics); //  resize(src, dst, size);
            }
       }


        // Save all weights
        if (save_cnt < save_after_nr)
        {
            save_cnt++;
        }
        else
        {
            save_cnt = 0;
            next_scene_fc_net.save_weights(next_scene_net_filename);
        }
        // End
    }
}

vector<int> fisher_yates_shuffle(vector<int> table)
{
    int size = table.size();
    for (int i = 0; i < size; i++)
    {
        table[i] = i;
    }
    for (int i = size - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = table[i];
        table[i] = table[j];
        table[j] = temp;
    }
    return table;
}
