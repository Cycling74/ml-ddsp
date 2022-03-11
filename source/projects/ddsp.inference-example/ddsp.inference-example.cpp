/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>
#include <regex>

using namespace c74::min;


class inference_example : public object<inference_example> {
public:
    MIN_DESCRIPTION	{"Run inference using libtorch."};
    MIN_TAGS		{"DDSP"};
    MIN_AUTHOR		{"Cycling '74"};
    MIN_RELATED		{""};

    // define inlets and outlets
    inlet<>  in_bang	    { this, "(bang) run inference" };
    inlet<>  in_model_path  { this, "(symbol) path to pre-trained torchscript model" };
    inlet<>  in_topk        { this, "(int) set the number of k largest elements to be send to the outlet" };
    outlet<> out_values	    { this, "(anything) output the resulting values of the inference" };
    outlet<> out_indices    { this, "(anything) output the resulting indices of the inference" };

    // define arguments
    argument<int> arg_topk { this, "topk", "Initial value for the number of k largest elements.",
        MIN_ARGUMENT_FUNCTION {
            m_topk = arg;
        }
    };
    
    // define attributes
    // TODO: integrate model path in attributes
    attribute<symbol> attr_model_path { this, "model_path", "",
        description {
            "Specify path to pre-trained torchscript model (*.pt)"
        }
    };
        
    // respond to int messages
    message<> msg_int { this, "int", "Set the number of k largest elements to be send to the outlet.",
        MIN_FUNCTION {
            switch(inlet) {
                case 2:
                    if ((int) args[0] < 1) {
                        m_topk = 1;
                    }
                    else {
                        m_topk = args[0];
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
    message<> msg_anything { this, "anything", "Set the path to th pre-trained model.",
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
                        m_module = torch::jit::load(formatted_path);
                    } catch (const c10::Error& e) {
                        m_module_is_loaded = false;
                        cerr << "Error loading model" << endl;
                        cerr << e.what_without_backtrace() << endl;
                        return {};
                    }
                    m_module_is_loaded = true;
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
            
            if (!m_module_is_loaded) {
                cout << "No valid model loaded" << endl;
                return {};
            }
            
            torch::NoGradGuard no_grad; // ensures that autograd is off
            
            m_module.eval(); // turn off dropout and other training-time layers/functions
            
            // create a dummy input
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(torch::rand({1, 3, 224, 224}));

            // execute model and package output as tensor
            at::Tensor model_output = m_module.forward(inputs).toTensor();

            namespace F = torch::nn::functional;
            at::Tensor output_sm = F::softmax(model_output, F::SoftmaxFuncOptions(1));
            
            // extract k largest elements (values + indices)
            std::tuple<at::Tensor, at::Tensor> topk_tensor = output_sm.topk(m_topk);
            at::Tensor topk_values = std::get<0>(topk_tensor);
            at::Tensor topk_indices = std::get<1>(topk_tensor);
            
            atoms values(m_topk);
            atoms indices(m_topk);
            float* values_arr = (float*)topk_values.data_ptr();
            int* indices_arr = (int*)topk_indices.data_ptr();
            for (int i = 0; i < m_topk; ++i)
            {
                values[i] = (*values_arr++);
                indices[i] = (*indices_arr++);
                indices_arr++;
            }

            // send model ouput to outlets
            out_values.send(values);
            out_indices.send(indices);
            return {};
        }
    };

    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "torch inference example loaded" << endl;
            return {};
        }
    };
    
private:
    symbol m_model_path = "";
    int m_topk = 5;
    
    torch::jit::script::Module m_module; // deserialize ScriptModule
    bool m_module_is_loaded = false;
};

MIN_EXTERNAL(inference_example);
