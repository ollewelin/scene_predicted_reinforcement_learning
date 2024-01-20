
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
#include <float.h>

#include "pinball_game.hpp"
#define MOVE_DOWN 0
#define MOVE_UP 1
#define MOVE_STOP 2
#define NORMAL_STATE 0
#define ONE_STEP_BEFORE_TERMINAL_STATE 1
#define NOW_TERMINAL_STATE 2

const int pixel_height = 35;                        /// The input data pixel height, note game_Width = 220
const int pixel_width = 35;                         /// The input data pixel width, note game_Height = 200


int do_dice_action(int number_of_actions)
{
    float max_decision = 0.0f;
    int decided_action = 0;
    for (int i = 0; i < number_of_actions; i++) // end_out_nodes = numbere of actions
    {
        float action_dice = (float)(rand() % 65535) / 65536; // Through a fair dice. Random value 0..1.0 range
        // cout << "action_dice = " << action_dice << endl;
        if (action_dice > (float)max_decision)
        {
            max_decision = (float)action_dice;
            decided_action = i;
        }
    }
    return decided_action;
}

vector<int> fisher_yates_shuffle(vector<int> table);

/*
// predict_next_frame pred_next_frame;
// call example
//pred_next_frame.make_and_store_predicted_frames(mat_next_scene_all_actions, next_scene_fc_net, pixel_height, pixel_width, act);

class predict_next_frame
{
    public:
    void make_and_store_predicted_frames(Mat& mat_next_scene_all_actions, fc_m_resnet& next_scene_fc_net, int pixel_height, int pixel_width, int action)
    {
        // Store all predictied next video frame for each diffrent action.
        for (int row = 0; row < pixel_height; row++)
        {
            for (int col = 0; col < mat_next_scene_all_actions.cols; col++)
            {
                // Store all predictied next video frame for each diffrent action.
                mat_next_scene_all_actions.at<float>(row + action * pixel_width, col) = next_scene_fc_net.output_layer[row * pixel_width + col];
            }
        }
    }
};
*/

typedef struct
{
    vector<double> video_frame; // vector of doubles to store the video frame
    int selected_action;        // integer to store the selected action
    int dice_used;              // integer to store if the dice was used = 1 or not used = 0
    double rewards_Q;           // store the rewards at real game. Then recalculate this before training with a decay factor
    // int terminal_state;         // 1 = we are in terminal state. 0 normal state
} replay_data_struct;

/*
class get_frames_from_replay_buf
{
    public:
    void load_replay_frames_to_policy_net(fc_m_resnet& policy_fc_net, vector<vector<replay_data_struct>>& replay_buffer, int g_replay_cnt, int frame_g, )
    {
        
        //check size
        int r_fr_size = replay_frame.size();
        if(r_fr_size == pixel_height * pixel_width)
        {
            int indx=0;
            for (int row = 0; row < pixel_height; row++)
            {
                for (int col = 0; col < pixel_width; col++)
                {
                    //   replay_buffer[g_replay_cnt][frame_g].video_frame[row * pixel_width + col] = resized_grapics.at<float>(row, col);
                    indx++;
                }
            }
        }
        else
        {
            cout << "Error replay_frame size" << endl;
        }
    }
};
*/
int main()
{
    int term_state = NORMAL_STATE;//0 = N
    int skip_scene_predicotr_only_for_benchmarking = 0; // set this to 1 to benchmaring with only use policy network like orderanry vanilla on policy reinforcemnat learning instead of dubble network with scenen predictor
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
    gameObj1.use_only_3_ball_angle_game = 0;

    // Set up a OpenCV mat
    Size image_size_reduced(pixel_height, pixel_width); // the dst image size,e.g.50x50
    Mat resized_grapics, game_video_full_size;
    const int nr_frames_strobed = 4; // 4 Images in serie to make neural network to see movments


    replay_data_struct replay_struct_item;
    replay_struct_item.selected_action = 0;
    replay_struct_item.dice_used = 0;
    replay_struct_item.rewards_Q = 0.0;
    replay_struct_item.video_frame.resize(pixel_height * pixel_width);

    vector<replay_data_struct> replay_1_episode_data_buffer;
    for (int i = 0; i < gameObj1.nr_of_frames; ++i)
    {
        replay_1_episode_data_buffer.push_back(replay_struct_item);
    }

    //=========== Neural Network size settings ==============
    fc_m_resnet next_scene_fc_net;
    fc_m_resnet policy_fc_net;
    int save_cnt = 0;

    string next_scene_net_filename;
    next_scene_net_filename = "next_scene_net_weights.dat";
    string policy_fc_net_filename;
    policy_fc_net_filename = "policy_fc_net_weights.dat";

    next_scene_fc_net.get_version();
    next_scene_fc_net.block_type = 2;
    next_scene_fc_net.use_softmax = 0;                         // 0= Not softmax for reinforcement learning
    next_scene_fc_net.activation_function_mode = 2;            // ReLU for all fully connected activation functions except output last layer
    next_scene_fc_net.force_last_activation_function_mode = 0; // 1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
    next_scene_fc_net.use_skip_connect_mode = 0;               // 1 for residual network architetcture
    next_scene_fc_net.use_dropouts = 0;
    next_scene_fc_net.dropout_proportion = 0.0;
    next_scene_fc_net.clip_deriv = 0;

    policy_fc_net.block_type = 2;
    policy_fc_net.use_softmax = 0;
    policy_fc_net.activation_function_mode = 2;
    policy_fc_net.force_last_activation_function_mode = 0;
    policy_fc_net.use_skip_connect_mode = 0;
    policy_fc_net.use_dropouts = 0;
    policy_fc_net.dropout_proportion = 0.0;
    policy_fc_net.clip_deriv = 0;

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

    const int policy_net_hid_layers = 3;
    const int policy_net_hid_nodes_L1 = 200;
    const int policy_net_hid_nodes_L2 = 50;
    const int policy_net_hid_nodes_L3 = 10;
    const int policy_net_out_nodes = nr_of_actions; //
    const int policy_net_inp_nodes = pixel_height * pixel_width * nr_frames_strobed + nr_of_actions;

    //---------- next scene fc net layer setup --------
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

    //-------- policy net layer setup -----
    for (int i = 0; i < policy_net_inp_nodes; i++)
    {
        policy_fc_net.input_layer.push_back(0.0);
        policy_fc_net.i_layer_delta.push_back(0.0);
    }

    for (int i = 0; i < policy_net_out_nodes; i++)
    {
        policy_fc_net.output_layer.push_back(0.0);
        policy_fc_net.target_layer.push_back(0.0);
    }
    policy_fc_net.set_nr_of_hidden_layers(next_scene_hid_layers);
    policy_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L1);
    policy_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L2);
    policy_fc_net.set_nr_of_hidden_nodes_on_layer_nr(next_scene_hid_nodes_L3);

    //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(end_hid_layers)

    typedef struct
    {
        int frame_index_of_rand_act;
        int episode_index_of_rand_act;
    } random_action_data_struct;
    random_action_data_struct rand_act_data_item;
    rand_act_data_item.frame_index_of_rand_act = 0;
    rand_act_data_item.episode_index_of_rand_act = 0;
    vector<random_action_data_struct> rand_action_list;
    rand_action_list.clear();
    for (int i = 0; i < 10; i++)
    {
        rand_action_list.push_back(rand_act_data_item);
    }

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
    const int g_replay_size = 100; // Should be 10000 or more
    const int save_after_nr = 1;
    // statistics report
    const int max_w_p_nr = 1000;
    int win_p_cnt = 0;
    int win_counter = 0;
    double last_win_probability = 0.5;
    double now_win_probability = last_win_probability;

    vector<vector<replay_data_struct>> replay_buffer;
    for (int i = 0; i < g_replay_size; i++)
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
        policy_fc_net.load_weights(policy_fc_net_filename);

    }
    else
    {
        next_scene_fc_net.randomize_weights(init_random_weight_propotion);
        policy_fc_net.randomize_weights(init_random_weight_propotion);
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
            gameObj1.move_up = do_dice_action(nr_of_actions);
            for (int frame_g = 0; frame_g < gameObj1.nr_of_frames; frame_g++) // Loop throue each of the 100 frames
            {
                gameObj1.frame = frame_g;
                gameObj1.run_episode();
                game_video_full_size = gameObj1.gameGrapics.clone();
                resize(game_video_full_size, resized_grapics, image_size_reduced);
                imshow("resized_grapics", resized_grapics); //  resize(src, dst, size);
                waitKey(1);
                for (int row = 0; row < pixel_height; row++)
                {
                    for (int col = 0; col < pixel_width; col++)
                    {
                        replay_buffer[g_replay_cnt][frame_g].video_frame[row * pixel_width + col] = resized_grapics.at<float>(row, col);
                    }
                }

                if (frame_g > nr_frames_strobed - 1) // Wait until all 4 images is up there in the game after start
                {
                   
                    if(frame_g < gameObj1.nr_of_frames - 1)
                    {
                        term_state = NORMAL_STATE;
                    }
                    if(frame_g == gameObj1.nr_of_frames - 1)
                    {
                        term_state = NOW_TERMINAL_STATE;
                    }
                    if(frame_g == gameObj1.nr_of_frames - 2)
                    {
                        term_state = ONE_STEP_BEFORE_TERMINAL_STATE;
                    }

                    int inp_n_idx = 0;
                    if(term_state != NOW_TERMINAL_STATE)
                    {
                        float exploring_dice = (float)(rand() % 65535) / 65536; // Through a fair dice. Random value 0..1.0 range
                        if (exploring_dice < epsilon)
                        {
                            // Choose dice action (Exploration mode)
                            gameObj1.move_up = do_dice_action(nr_of_actions);
                            replay_buffer[g_replay_cnt][frame_g].dice_used = 1;
                        }
                        else
                        {
                            // Choose predicted action (Exploit mode)
                            //
                            for (int f = 0; f < nr_frames_strobed; f++)
                            {
                                for (int pix_idx = 0; pix_idx < (pixel_width * pixel_height); pix_idx++)
                                {
                                    double pixel_d = (double)replay_buffer[g_replay_cnt][frame_g - nr_frames_strobed + f].video_frame[pix_idx];

                                    next_scene_fc_net.input_layer[inp_n_idx] = pixel_d;
                                    policy_fc_net.input_layer[inp_n_idx] = pixel_d;
                                    inp_n_idx++;
                                }
                            }
                            double strongest_action_value = -DBL_MAX;
                            double sum_up_all_action_value_at_term_state = 0;
                            int which_next_frame_have_stongest_action = 0;
                            for (int act = 0; act < nr_of_actions; act++)
                            {
                                // Loop thorugh all possible actont and try to predict next scene on each taken action.
                                for (int i = 0; i < nr_of_actions; i++)
                                {
                                    double one_hot_encode_action_input_node = 0.0;
                                    if (i == act)
                                    {
                                        one_hot_encode_action_input_node = 1.0;
                                    }
                                    else
                                    {
                                        one_hot_encode_action_input_node = 0.0;
                                    }
                                    next_scene_fc_net.input_layer[inp_n_idx + i] = one_hot_encode_action_input_node; // One hot encoding
                                }
                                next_scene_fc_net.forward_pass(); // Do one prediction of next video frame how it will looks on one singel spefific action taken.

                                // Store all predictied next video frame for each diffrent action.
                                for (int row = 0; row < pixel_height; row++)
                                {
                                    for (int col = 0; col < mat_next_scene_all_actions.cols; col++)
                                    {
                                        // Store all predictied next video frame for each diffrent action.
                                        double next_sc_pix = next_scene_fc_net.output_layer[row * pixel_width + col];
                                        mat_next_scene_all_actions.at<float>(row + act * pixel_width, col) = next_sc_pix;
                                        policy_fc_net.input_layer[row * pixel_width + col] = next_sc_pix;
                                    }
                                }

                                // Run through prediced frames to the policy network and check it this predicted frame will give the policy net a strongest positive next action
                                policy_fc_net.forward_pass();
                                // int what_act_was_stongest_i_debug = 0;
                                // double debug_v = 0;
                                for (int i = 0; i < nr_of_actions; i++)
                                {
                                    double action_policy_net_output = policy_fc_net.output_layer[i];
                                    if (term_state == NORMAL_STATE)
                                    {
                                        if (action_policy_net_output > strongest_action_value)
                                        {
                                            strongest_action_value = action_policy_net_output; // Store the strongest policy value
                                            which_next_frame_have_stongest_action = act;       // Store what action prediced frame of the next predicted frame have the strongest action value
                                            // what_act_was_stongest_i_debug = i;
                                            // debug_v = strongest_action_value;
                                        }
                                    }
                                    else
                                    {
                                        // ONE_STEP_BEFORE_TERMINAL_STATE
                                        if (i == 0)
                                        {
                                            sum_up_all_action_value_at_term_state = 0;
                                        }
                                        // At the predicted terminal state use summed action value instead of the max next action value
                                        // because in the terminal state action don't matters for the reward (no more action is possible to take at terminal)
                                        // so at termninal state the policy net will be train on same rewards value on ALL output action node as a target
                                        // therefor a sum up of all action is a convinient way to use the policy network i think
                                        sum_up_all_action_value_at_term_state += action_policy_net_output;
                                        if (i == nr_of_actions - 1)
                                        {
                                            // Check agianst the other next predicted frame summed action values
                                            if (sum_up_all_action_value_at_term_state > strongest_action_value)
                                            {
                                                which_next_frame_have_stongest_action = act;
                                            }
                                        }
                                    }
                                }
                                // cout << "which_next_frame_have_stongest_action = " << which_next_frame_have_stongest_action << " what_act_was_stongest_i_debug = " <<  what_act_was_stongest_i_debug << " debug_v = " << debug_v << endl;
                                // Show next frame prediction
                                imshow("mat_next_scene_all_actions", mat_next_scene_all_actions);
                                waitKey(1);
                                gameObj1.move_up = which_next_frame_have_stongest_action;
                                replay_buffer[g_replay_cnt][frame_g].dice_used = 0;
                            }
                            
                        }
                        replay_buffer[g_replay_cnt][frame_g].rewards_Q = 0.0;

                    }
                    else
                    {
                        //Ternimal state
                        //Get the terminal rewards

                        //double abs_diff = abs(gameObj1.pad_ball_diff);
                        // cout << "pad_ball_diff = " << abs_diff << endl;
                        //if (abs_diff < 1.0)
                        //{
                        //    abs_diff = 1.0;
                        //}

                        double rewards = 0.0;
                        if (gameObj1.win_this_game == 1)
                        {
                            if (gameObj1.square == 1)
                            {

                                rewards = reward_gain * 1.0; // Win Rewards avoid square
                                                             //       rewards /= abs_diff;
                            }
                            else
                            {
                                rewards = reward_gain * 1.0; // Win Rewards catch ball
                                                             //       rewards /= abs_diff;
                            }
                            win_counter++;
                        }
                        else
                        {
                            if (gameObj1.square == 1)
                            {
                                //  rewards = -2.35; // Lose Penalty
                                rewards = reward_gain * (-1.0);
                                // rewards /= abs_diff;
                            }
                            else
                            {
                                // rewards = -3.95; // Lose Penalty
                                rewards = reward_gain * (-1.0);
                                // rewards *= abs_diff;
                            }
                        }
                        replay_buffer[g_replay_cnt][frame_g].rewards_Q = rewards;
                        cout << "                                                                                                       " << endl;
                        std::cout << "\033[F";
                        cout << "Play g_replay_cnt = " << g_replay_cnt << " Rewards = " << rewards << endl;
                        // Move the cursor up one line (ANSI escape code)
                        std::cout << "\033[F";
                    }
                }
                else
                {
                    //Before the first frame in series filled up
                    gameObj1.move_up = do_dice_action(nr_of_actions);
                    replay_buffer[g_replay_cnt][frame_g].dice_used = 1;
                }
                replay_buffer[g_replay_cnt][frame_g].selected_action = gameObj1.move_up;
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
