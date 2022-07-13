{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 2,
			"revision" : 2,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 34.0, 90.0, 750.0, 728.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-55",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 72.0, 1003.085585594177246, 77.0, 33.0 ],
					"presentation_linecount" : 2,
					"text" : "enable / disable DSP"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-56",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 124.0, 379.0, 31.0, 22.0 ],
					"text" : "load"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-54",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 367.514999499999931, 702.5, 216.0, 20.0 ],
					"text" : "click to load learned impulse response"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-53",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 207.0, 702.5, 141.0, 20.0 ],
					"text" : "double click to open GUI"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-52",
					"linecount" : 2,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 367.514999499999931, 659.5, 137.0, 35.0 ],
					"text" : ";\rconvolution-reverb iradd"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "signal", "signal", "" ],
					"patching_rect" : [ 159.0, 672.5, 189.0, 22.0 ],
					"text" : "hirt.convolver~ convolution-reverb",
					"varname" : "hirt.convolver~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-49",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 386.014999499999931, 288.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-50",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 159.014999499999931, 288.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-46",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1230.985000500000069, 288.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-47",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1040.985000500000069, 288.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-43",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1040.985000500000069, 225.585585594177246, 32.0, 22.0 ],
					"text" : "mtof"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-44",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 1230.0, 259.0, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-45",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 1040.985000500000069, 259.0, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-41",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 854.0, 370.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 702.0, 370.5, 150.0, 20.0 ],
					"text" : "connect to decoder inlet 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 101.0, 673.5, 48.0, 20.0 ],
					"text" : "Reverb"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-38",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 407.014999499999931, 190.5, 58.0, 20.0 ],
					"text" : "loudness"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-37",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 256.014999499999931, 248.0, 52.0, 20.0 ],
					"text" : "f0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 442.014999499999931, 123.0, 58.0, 20.0 ],
					"text" : "loudness"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-28",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 215.014999499999931, 123.0, 52.0, 20.0 ],
					"text" : "f0"
				}

			}
, 			{
				"box" : 				{
					"bottomvalue" : -20,
					"id" : "obj-25",
					"maxclass" : "pictslider",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "int", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 159.014999499999931, 160.5, 246.0, 80.0 ],
					"rightvalue" : 2000,
					"topvalue" : 200
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-21",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 159.0, 69.0, 173.0, 33.0 ],
					"presentation_linecount" : 2,
					"text" : "OPTION 1:\nParameter Space Exploration"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1041.0, 67.0, 89.0, 33.0 ],
					"presentation_linecount" : 2,
					"text" : "OPTION 3:\nMIDI Control"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 593.0, 69.0, 98.0, 33.0 ],
					"presentation_linecount" : 3,
					"text" : "OPTION 2:\nTimbre Transfer"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-29",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 159.0, 495.0, 166.0, 22.0 ],
					"text" : "ddsp.mc-harmonic-oscillator~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 159.0, 541.0, 92.0, 22.0 ],
					"text" : "mc.mixdown~ 1"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-42",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 384.514999499999931, 122.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-39",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 159.014999499999931, 122.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-35",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 386.014999499999931, 258.5, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-31",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 159.014999499999931, 258.5, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-36",
					"maxclass" : "kslider",
					"mode" : 2,
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "int", "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1040.985000500000069, 109.0, 336.0, 53.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-33",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 159.0, 610.0, 29.5, 22.0 ],
					"text" : "+~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-32",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 367.514999499999931, 495.0, 114.0, 22.0 ],
					"text" : "ddsp.filtered-noise~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 593.0, 160.5, 98.0, 33.0 ],
					"text" : "Pitch / loudness tracking"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 353.585585594177246, 135.0, 20.0 ],
					"text" : "load control model (*.ts)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 784.0, 68.0, 89.0, 20.0 ],
					"text" : "load audio file"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 854.0, 305.0, 106.0, 22.0 ],
					"text" : "onepole~ 1000 Hz"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 702.0, 305.0, 106.0, 22.0 ],
					"text" : "onepole~ 1000 Hz"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 64.0, 872.0, 85.0, 20.0 ],
					"presentation_linecount" : 3,
					"text" : "unmute / mute"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 125.0, 897.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 159.0, 945.0, 39.0, 22.0 ],
					"text" : "gate~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 159.0, 991.085585594177246, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-22",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 2,
					"numoutlets" : 5,
					"outlettype" : [ "signal", "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 159.0, 785.0, 48.0, 136.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_longname" : "live.gain~",
							"parameter_mmax" : 6.0,
							"parameter_mmin" : -70.0,
							"parameter_shortname" : "live.gain~",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "live.gain~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 732.0, 68.0, 35.0, 22.0 ],
					"text" : "open"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 702.0, 67.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-48",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "bang" ],
					"patching_rect" : [ 702.0, 108.085585594177246, 95.0, 22.0 ],
					"text" : "sfplay~ @loop 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 159.0, 739.085585594177246, 37.0, 22.0 ],
					"text" : "*~ 30"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 702.0, 225.585585594177246, 32.0, 22.0 ],
					"text" : "mtof"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "multichannelsignal", "multichannelsignal", "multichannelsignal" ],
					"patching_rect" : [ 159.0, 436.085585594177246, 128.0, 22.0 ],
					"text" : "ddsp.control-decoder~"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-10",
					"maxclass" : "number~",
					"mode" : 2,
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "float" ],
					"patching_rect" : [ 854.0, 341.5, 56.0, 22.0 ],
					"sig" : 0.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 854.0, 259.0, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontname" : "Arial",
					"fontsize" : 12.0,
					"id" : "obj-9",
					"maxclass" : "number~",
					"mode" : 2,
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "float" ],
					"patching_rect" : [ 702.0, 341.5, 56.0, 22.0 ],
					"sig" : 0.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 702.0, 259.0, 31.0, 22.0 ],
					"text" : "sig~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 702.0, 160.5, 171.0, 22.0 ],
					"text" : "sigmund~ @hop 512 pitch env"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-29", 1 ],
					"source" : [ "obj-1", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-29", 0 ],
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-32", 0 ],
					"source" : [ "obj-1", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"source" : [ "obj-11", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 1 ],
					"order" : 0,
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 0 ],
					"order" : 1,
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-48", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"midpoints" : [ 134.5, 940.0, 168.5, 940.0 ],
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-48", 0 ],
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 0 ],
					"source" : [ "obj-2", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 1 ],
					"midpoints" : [ 168.5, 934.0, 188.5, 934.0 ],
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-31", 0 ],
					"source" : [ "obj-25", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 0 ],
					"source" : [ "obj-25", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"source" : [ "obj-29", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-34", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-31", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 1 ],
					"source" : [ "obj-32", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-33", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 1 ],
					"order" : 0,
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 0 ],
					"order" : 1,
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 1 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 0 ],
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-44", 0 ],
					"source" : [ "obj-36", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 0 ],
					"source" : [ "obj-39", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 1 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 0 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-2", 0 ],
					"source" : [ "obj-48", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-56", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-9", 0 ],
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-22" : [ "live.gain~", "live.gain~", 0 ],
			"obj-3::obj-108::obj-90" : [ "number[2]", "number", 0 ],
			"obj-3::obj-10::obj-17::obj-70" : [ "hirt.val[14]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-19::obj-70" : [ "hirt.val[13]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-21" : [ "Saturation Type", "Saturation", 0 ],
			"obj-3::obj-10::obj-22::obj-70" : [ "hirt.val[12]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-3" : [ "EQ Routing", "EQ", 0 ],
			"obj-3::obj-10::obj-50::obj-70" : [ "hirt.val[11]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-51::obj-70" : [ "hirt.val[10]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-53::obj-70" : [ "hirt.val[9]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-54::obj-70" : [ "hirt.val[8]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-55::obj-70" : [ "hirt.val[7]", "hirt.val", 0 ],
			"obj-3::obj-10::obj-56::obj-70" : [ "hirt.val[6]", "hirt.val", 0 ],
			"obj-3::obj-1::obj-48::obj-70" : [ "hirt.val[1]", "hirt.val", 0 ],
			"obj-3::obj-1::obj-59::obj-70" : [ "hirt.val[2]", "hirt.val", 0 ],
			"obj-3::obj-1::obj-97::obj-70" : [ "hirt.val", "hirt.val", 0 ],
			"obj-3::obj-28" : [ "PATCH/PRESS", "PATCH/PRESS", 0 ],
			"obj-3::obj-36::obj-131::obj-10" : [ "Category Menu", "Category Menu", 0 ],
			"obj-3::obj-36::obj-131::obj-11" : [ "IR Menu", "IR Menu", 0 ],
			"obj-3::obj-36::obj-35" : [ "Drop IR", "live.drop", 3 ],
			"obj-3::obj-3::obj-63::obj-70" : [ "hirt.val[3]", "hirt.val", 0 ],
			"obj-3::obj-3::obj-64::obj-70" : [ "hirt.val[4]", "hirt.val", 0 ],
			"obj-3::obj-3::obj-65::obj-70" : [ "hirt.val[5]", "hirt.val", 0 ],
			"parameterbanks" : 			{

			}
,
			"parameter_overrides" : 			{
				"obj-3::obj-10::obj-17::obj-70" : 				{
					"parameter_exponent" : 3.0,
					"parameter_initial" : 0.5,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[14]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.05, 18.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 10
				}
,
				"obj-3::obj-10::obj-19::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 0.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[13]",
					"parameter_modmode" : 0,
					"parameter_range" : [ -18.0, 18.0 ],
					"parameter_type" : 0,
					"parameter_units" : " ",
					"parameter_unitstyle" : 4
				}
,
				"obj-3::obj-10::obj-22::obj-70" : 				{
					"parameter_exponent" : 4.0,
					"parameter_initial" : 125.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[12]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 10.0, 18000.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 3
				}
,
				"obj-3::obj-10::obj-50::obj-70" : 				{
					"parameter_exponent" : 3.0,
					"parameter_initial" : 0.707107,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[11]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.05, 18.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 10
				}
,
				"obj-3::obj-10::obj-51::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 0.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[10]",
					"parameter_modmode" : 0,
					"parameter_range" : [ -18.0, 18.0 ],
					"parameter_type" : 0,
					"parameter_units" : " ",
					"parameter_unitstyle" : 4
				}
,
				"obj-3::obj-10::obj-53::obj-70" : 				{
					"parameter_exponent" : 4.0,
					"parameter_initial" : 1000.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[9]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 10.0, 18000.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 3
				}
,
				"obj-3::obj-10::obj-54::obj-70" : 				{
					"parameter_exponent" : 3.0,
					"parameter_initial" : 0.5,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[8]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.05, 18.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 10
				}
,
				"obj-3::obj-10::obj-55::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 0.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[7]",
					"parameter_modmode" : 0,
					"parameter_range" : [ -18.0, 18.0 ],
					"parameter_type" : 0,
					"parameter_units" : " ",
					"parameter_unitstyle" : 4
				}
,
				"obj-3::obj-10::obj-56::obj-70" : 				{
					"parameter_exponent" : 4.0,
					"parameter_initial" : 8000.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[6]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 10.0, 18000.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 3
				}
,
				"obj-3::obj-1::obj-48::obj-70" : 				{
					"parameter_exponent" : 1.58,
					"parameter_initial" : 100.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 1,
					"parameter_longname" : "hirt.val[1]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 50.0, 200.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 5
				}
,
				"obj-3::obj-1::obj-59::obj-70" : 				{
					"parameter_exponent" : 2.0,
					"parameter_initial" : 0.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[2]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.0, 200.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 2
				}
,
				"obj-3::obj-1::obj-97::obj-70" : 				{
					"parameter_exponent" : 1.01,
					"parameter_initial" : 100.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 1,
					"parameter_modmode" : 0,
					"parameter_range" : [ 1.0, 200.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 5
				}
,
				"obj-3::obj-3::obj-63::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 100.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[3]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.0, 100.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 5
				}
,
				"obj-3::obj-3::obj-64::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 0.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[4]",
					"parameter_modmode" : 0,
					"parameter_range" : [ -20.0, 20.0 ],
					"parameter_type" : 0,
					"parameter_units" : " ",
					"parameter_unitstyle" : 4
				}
,
				"obj-3::obj-3::obj-65::obj-70" : 				{
					"parameter_exponent" : 1.0,
					"parameter_initial" : 50.0,
					"parameter_initial_enable" : 0,
					"parameter_invisible" : 0,
					"parameter_longname" : "hirt.val[5]",
					"parameter_modmode" : 0,
					"parameter_range" : [ 0.0, 100.0 ],
					"parameter_type" : 0,
					"parameter_unitstyle" : 5
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "HIRT_HISSTools_Logo.png",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/misc/HIRT_image",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/misc/HIRT_image",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "bufresample~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "ddsp.control-decoder~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "ddsp.filtered-noise~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "ddsp.mc-harmonic-oscillator~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "hirt.convolver.realtime~.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.convolver~.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.convolvestereo~.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.dial.linear.only.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.dial.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.size.resample.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt.svfcoeff.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_base_name.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_conv_zoom.txt",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_convolution_rt_library.genexpr",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/code",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/code",
				"type" : "GenX",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_convolver_clientlist_alias.txt",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_convolver_rt_part1.gendsp",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/code",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/code",
				"type" : "gDSP",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_convolver_rt_part3.gendsp",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/code",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/code",
				"type" : "gDSP",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_cv_info_view.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_data_colls.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_decay_size_pre.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_eq_library.genexpr",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/code",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/code",
				"type" : "GenX",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_eq_sat.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_file_check.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_file_loading.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_file_picker.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_file_set.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_filter_type.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_folder.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_gain_display_single.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_gain_params.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_reverb/HIRT_reverb_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_reverb/HIRT_reverb_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_interface_eq.js",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/jsui",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/jsui",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_ir_single.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_loading_scheme.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_nan_fix.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_output_mini.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_convolver/HIRT_convolver_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_size_resample_feed.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_reverb/HIRT_reverb_support",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/patchers/HIRT_reverb/HIRT_reverb_support",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "hirt_zoom_factor.js",
				"bootpath" : "~/Documents/Max 8/Packages/HISSTools Impulse Response Toolbox (HIRT)/javascript",
				"patcherrelativepath" : "../../HISSTools Impulse Response Toolbox (HIRT)/javascript",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "iraverage~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "irdisplay~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "irstats~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "morphfilter~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "multiconvolve~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "sigmund~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "spectrumdraw~.mxo",
				"type" : "iLaX"
			}
 ],
		"autosave" : 0
	}

}
