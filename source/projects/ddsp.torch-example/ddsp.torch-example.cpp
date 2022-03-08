/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>

using namespace c74::min;


class torch_example : public object<torch_example> {
public:
    MIN_DESCRIPTION	{"Generate a random 2D-tensor."};
    MIN_TAGS		{"DDSP"};
    MIN_AUTHOR		{"Cycling '74"};
    MIN_RELATED		{""};

    // define inlets and outlets
    inlet<>  bang_in	{ this, "(bang) generate a random 2D-tensor" };
    inlet<>  num_rows_in  { this, "(int) change the number of rows of the tensor (greater than or equal to 1)" };
    inlet<>  num_columns_in  { this, "(int) change the number of columns of the tensor (greater than or equal to 1)" };
    outlet<> output	{ this, "(anything) output the generated tensor" };

    // define arguments
    argument<int> num_rows_arg { this, "num_rows", "Initial value for number of rows (greater than or equal to 1).",
        MIN_ARGUMENT_FUNCTION {
            m_num_rows = arg;
        }
    };
    
    argument<int> num_columns_arg { this, "num_columns", "Initial value for number of columns (greater than or equal to 1).",
        MIN_ARGUMENT_FUNCTION {
            m_num_columns = arg;
        }
    };
        
    // respond to int messages
    message<> msg_num_rows { this, "int", "Set the tensor dimensions.",
        MIN_FUNCTION {
            switch(inlet) {
                case 1:
                    if ((int) args[0] < 1) {
                        m_num_rows = 1;
                    }
                    else {
                        m_num_rows = args[0];
                    }
                    break;
                case 2:
                    if ((int) args[0] < 1) {
                        m_num_columns = 1;
                    }
                    else {
                        m_num_columns = args[0];
                    }
                    break;
                default:
                    assert(false);
                    break;
            }
            return {};
        }
    };

    // respond to bang message
    message<> bang { this, "bang", "Generate a random tensor and send to outlet.",
        MIN_FUNCTION {
            atoms result(m_num_rows*m_num_columns);
            torch::Tensor tensor = torch::rand({m_num_rows, m_num_columns}).to(torch::kFloat);
            float* temp_arr = (float*)tensor.data_ptr();
            for (int y = 0; y < tensor.sizes()[0]; ++y)
            {
                for (int x = 0; x < tensor.sizes()[1]; ++x)
                {
                    result[m_num_columns*y+x] = (*temp_arr++);
                }
            }
                
            cout << tensor << endl;    // post to the max console
            output.send(result);       // send to outlet
            return {};
        }
    };

    // when object is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "torch example loaded" << endl;
            return {};
        }
    };
    
private:
    int m_num_rows = 2;
    int m_num_columns = 3;
};

MIN_EXTERNAL(torch_example);
