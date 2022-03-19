/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>
#include <regex>

#define DEVICE torch::kCPU

using namespace c74::min;


class ddsp_decoder : public object<ddsp_decoder> {
public:
    MIN_DESCRIPTION	{"Standard DDSP decoder."};
    MIN_TAGS		{"DDSP"};
    MIN_AUTHOR		{"Cycling '74"};
    MIN_RELATED		{""};

    // define inlets and outlets
    inlet<>  in_bang	    { this, "(bang) run inference" };
    inlet<>  in_model_path  { this, "(symbol) path to pre-trained torchscript model" };
    inlet<>  in_pitch       { this, "(float) pitch" };
    inlet<>  in_loudness    { this, "(float) loudness" };
    outlet<> out_values	    { this, "(anything) output the resulting values of the inference" };
    outlet<> out_indices    { this, "(anything) output the resulting indices of the inference" };

    
    // define attributes
    // TODO: integrate model path in attributes
    attribute<symbol> attr_model_path { this, "model_path", "",
        description {
            "Specify path to pre-trained torchscript model (*.pt)"
        }
    };
        
    // respond to float messages
    message<> msg_float { this, "float", "Set pitch and loudness.",
        MIN_FUNCTION {
            switch(inlet) {
                case 2:
                    if ((float) args[0] < 1) {
                        m_pitch = 1.0f;
                    }
                    else {
                        m_pitch = args[0];
                    }
                    break;
                case 3:
                    if ((float) args[0] < 0) {
                        m_loudness = 0.0f;
                    }
                    else {
                        m_loudness = args[0];
                    }
                    break;
                default:
                    assert(false);
                    break;
            }
            return {};
        }
    };
    
    // respond to symbol messages
    message<> msg_anything { this, "anything", "Set the path to the pre-trained model.",
        MIN_FUNCTION {
            
            // convert the path received by Max to a C++ readable path
            string path = args[0];
            std::regex regex(":");
            std::vector<std::string> segmented_path(
                            std::sregex_token_iterator(path.begin(), path.end(), regex, -1),
                            std::sregex_token_iterator()
                            );
            string formatted_path = segmented_path[1];

            // load model
            switch(inlet) {
                case 1:
                    cout << "Loading model..." << endl;
                    
                    try {
                        m_scripted_model = torch::jit::load(formatted_path);
                    } catch (const c10::Error& e) {
                        m_model_is_loaded = false;
                        cerr << "Error loading model" << endl;
                        cerr << e.what_without_backtrace() << endl;
                        return {};
                    }
                    m_model_is_loaded = true;
                    m_model_path = formatted_path;

                    cout << "Model loaded successfully" << endl;

                    break;
                default:
                    assert(false);
                    break;
            }
            return {};
        }
    };

    // respond to bang message
    message<> msg_bang { this, "bang", "Generate a random tensor and send to outlet.",
        MIN_FUNCTION {
            
            cout << "bang msg" << endl;
            
            if (!m_model_is_loaded) {
                cout << "No valid model loaded" << endl;
                return {};
            }

            torch::NoGradGuard no_grad; // ensures that autograd is off

            m_scripted_model.eval(); // turn off dropout and other training-time layers/functions

            // input tensor (dummy input)

            at::Tensor pitch = torch::rand({1, 100, 1});
            at::Tensor loudness = torch::rand({1, 100, 1});

            std::vector<torch::jit::IValue> inputs = {pitch, loudness};

            cout << inputs << endl;

            // execute model and package output as tensor
            at::Tensor model_output = m_scripted_model.forward(inputs).toTensor();

            cout << model_output.sizes() << endl;

            // send model ouput to outlets
//            out_values.send(harmonic_amplitudes);
//            out_indices.send(transfer_function);
            return {};
        }
    };

    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "DDSP decoder loaded" << endl;
            return {};
        }
    };
    
private:
    symbol m_model_path = "";
    float m_pitch = 440.0f;
    float m_loudness = 1.0f;
    //float m_pitch_buffer[4] = {440.0f, 440.0f, 440.0f, 440.0f};
    //float m_loudness_buffer[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    torch::jit::script::Module m_scripted_model;
    bool m_model_is_loaded = false;
    //int m_buffer_size = 512;
};

MIN_EXTERNAL(ddsp_decoder);
