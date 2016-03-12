//////////////// Sequence //////////////////
////////////////////////////////////////////

function Sequence() {
	// create a new Parameters object
	var params = new Parameters();
    this.params = params

    // add the empty HTML table to the Sequence field (parameters will be populated into rows of the table once they are known, later)
    $("#seqparams").append(this.params.obj);

    this._handle_chgen = function() {
        $.getJSON("/ajax/gen_info/"+this.value+"/", 
        	{}, 
        	function(info) {
            	params.update(info.params);
        	}
        );
    }
    $("#seqgen").change(this._handle_chgen);

    $("#seqparams").click(
        // if you click on the parameters table and the drop-down list of 
        // available sequences (not generators) is enabled, then enable editing
    	function() {
	        if ($("#seqlist").attr("disabled") != "disabled") 
	            this.edit();
    	}.bind(this)
    );
    this.options = {};
}

Sequence.prototype.update = function(info) {
    console.log("Sequence.prototype.update");

    $("#seqlist").unbind("change");
    for (var id in this.options)
        $(this.options[id]).remove()
    if (document.getElementById("seqlist").tagName.toLowerCase() == "input")
        $("#seqlist").replaceWith("<select id='seqlist' name='seq_name'><option value='new'>Create New...</option></select>");
    
    // populate the list of available sequences already in the database
    this.options = {};
    var opt, id;
    for (id in info) {
        opt = document.createElement("option");
        opt.innerHTML = info[id].name;
        opt.value = id;

        this.options[id] = opt;
        $("#seqlist").append(opt);
    }

    if (id) { // TODO this is hacky.. 'id' is only true if info is defined/non-empty, check info more explicitly
        $("#seqgen option").each(function() {
            this.selected = false;
            if (this.value == info[id].generator[0])
                this.selected = true;
        })
        $("#seqlist option").each(function() {
            if (this.value == id)
                this.selected = true;
        })
        this.params.update(info[id].params);
        $("#seqstatic").attr("checked", info[id].static);

        //Bind the sequence list updating function
        var seq_obj = this;
        this._handle_chlist = function () {
            var id = this.value; // 'this' is bound to the options list when it's used inside the callback below
            if (id == "new")
                seq_obj.edit()
            else {
            	// the selected sequence is a previously used sequence, so populate the parameters from the db
                seq_obj.params.update(info[id].params);

                // disable editing in the table
                $("#seqparams input").attr("disabled", "disabled");  

                // change the value of the generator drop-down list to the generator for this sequence.
                $('#seqgen').val(info[id].generator[0]);

                // mark the static checkbox, if the sequence was static
                $("#seqstatic").attr("checked", info[id].static);
            }
        };
        $("#seqlist").change(this._handle_chlist);

        $("#seqstatic,#seqparams input, #seqgen").attr("disabled", "disabled");
    } else {
        this.edit();
        $("#seqgen").change();
    }
}

Sequence.prototype.destroy = function() {
	// clear out the 'options' dictionary
    for (var id in this.options)
        $(this.options[id]).remove()
    if (this.params) {
        $(this.params.obj).remove();  // remove the HTML table with the parameters in it

        delete this.params; // delete the JS object
    }
    $("#seqlist").unbind("change");
    $("#seqgen").unbind("change");
    if (document.getElementById("seqlist").tagName.toLowerCase() == "input")
        $("#seqlist").replaceWith("<select id='seqlist' name='seq_name'><option value='new'>Create New...</option></select>");
}

Sequence.prototype._make_name = function() {
    var gen = $("#sequence #seqgen option").filter(":selected").text()
    var txt = [];
    var d = new Date();
    var datestr =  d.getFullYear()+"."+(d.getMonth()+1)+"."+d.getDate()+" ";

    $("#sequence #seqparams input").each(function() { txt.push(this.name+"="+this.value); })
    return gen+":["+txt.join(", ")+"]"
}

Sequence.prototype.edit = function() {
    var _this = this;
    var curname = this._make_name();
    $("#seqlist").replaceWith("<input id='seqlist' name='seq_name' type='text' value='"+curname+"' />");
    $("#seqgen, #seqparams input, #seqstatic").removeAttr("disabled");
    
    this._handle_setname = function() { $
    	("#seqlist").attr("value", _this._make_name()); 
    };
    this._handle_chgen = function() {
        _this._handle_setname();
        $("#seqparams input").bind("blur.setname", _this._handle_setname );
    };
    $("#seqgen").change(this._handle_chgen);
    $("#seqparams input").bind("blur.setname", _this._handle_setname );
    this._handle_blurlist = function() {
        if (this.value != _this._make_name())
            $("#seqparams input").unbind("blur.setname", _this._handle_setname);
    };
    $("#seqlist").blur(this._handle_blurlist);
}


Sequence.prototype.enable = function() {
    $("#seqlist").removeAttr("disabled");
}
Sequence.prototype.disable = function() {
    $("#seqlist, #seqparams input, #seqgen, #seqstatic").attr("disabled", "disabled");
}

Sequence.prototype.get_data = function() {
	// Get data describing the sequence/generator parameters, to be POSTed to the webserver
    if ($("#sequence #seqlist").get(0).tagName == "INPUT") {
        //This is a new sequence, create new!
        var data = {};
        data['name'] = $("#seqlist").attr("value");
        data['generator'] = $("#seqgen").attr("value");
        data['params'] = this.params.to_json();
        data['static'] = $("#seqstatic").attr("checked") == "checked";
        return data;
    } else {
    	return parseInt($("#sequence #seqlist").attr("value"));	
    }
}




////////////////////// Report //////////////////////////
////////////////////////////////////////////////////////
function sec_to_time(len) {
    var d = new Date(len*1000);
    return d.toUTCString().slice(-12, -4);
}
var box_filters = {
    "state":function(str) { return str.slice(0,1).toUpperCase() + str.slice(1).toLowerCase(); },
    "length": sec_to_time,
    "reward_len": function(reward) { return sec_to_time(reward[0]) + " / "+reward[1]; },
    "rates":function(rates) {
        return (rates[0]*100).toPrecision(3) + "% / "+ (rates[1]*100).toPrecision(3) + "%";
    }
}

// Report class constructor
function Report(callback) {
    this.notify = callback;
    this.obj = document.createElement("div");
    this.info = document.createElement("table");
    this.msgs = document.createElement("div");
    this.stdout = document.createElement("pre");
    this.info.className = "options";
    this.msgs.className = "rightside";

    var linebreak = document.createElement("div");
    linebreak.className = "clear";

    this.obj.appendChild(this.info);
    this.obj.appendChild(linebreak);
    this.obj.appendChild(this.msgs);

    $("#report").append(this.obj);

    this.boxes = {};
}

Report.prototype.activate = function() {
	console.log('activating report');

    if (!this.ws) { 
        // Create a new JS WebSocket object
        this.ws = new WebSocket("ws://"+hostname.split(":")[0]+":8001/connect");

        this.ws.onmessage = function(evt) {
            //console.log(evt.data);
            var report = JSON.parse(evt.data);
            if (this.infos)
                this.infos.push(report)
            else
                this.update(report);
        }.bind(this);
    }
}

Report.prototype.deactivate = function() {
    /*
        Close the report websocket 
    */
    if (this.ws)
        this.ws.close();
    delete this.ws;
}

Report.prototype.update = function(info) {
    if (typeof(this.notify) == "function" && info)
        this.notify(info);
    if (info.status && info.status == "error") { // received an error message through the websocket
    	// append the error message (pre-formatted by python traceback) onto the printed out messages
        this.msgs.innerHTML += "<pre>"+info.msg+"</pre>";
    } else if (info.status && info.status == "stdout") {
        if (!this.stdout.parentNode)
            this.msgs.appendChild(this.stdout)

        this.stdout.innerHTML += info.msg;
    } else {
        for (var stat in info) {
			if (!this.boxes[stat]) { // if we haven't already made a table row for this stat
				if (!stat.match("status|task|subj|date|idx")) { // if this is not one of the stats we ignore because it's reported elsewhere
	                var row = document.createElement("tr");

	            	// make a column in the row for the stat name
	                var label = document.createElement("td");
	                label.innerHTML = stat;
	                
	                // make a column in the row to hold the data to be updated by the server
	                var data = document.createElement("td");

	                row.appendChild(label);
	                row.appendChild(data);

	                this.info.appendChild(row);

	                // save ref to the 'data' box, to be updated when new 'info' comes in
	                this.boxes[stat] = data;					
				}
            }
        }

        // Update the stat data
        for (var stat in this.boxes) {
            if (info[stat])
                if (box_filters[stat]){
                    console.log("Calling box filter for stat ", stat);
                    this.boxes[stat].innerHTML = box_filters[stat](info[stat]);
                }
                else
                    this.boxes[stat].innerHTML = info[stat];
        }
    }
}
Report.prototype.destroy = function () {
    this.deactivate();
    if (this.obj.parentNode)
        this.obj.parentNode.removeChild(this.obj);
}


// These functions are unused
Report.prototype.pause = function() {
    this.infos = [];
}
Report.prototype.unpause = function() {
    if (this.infos.length > 0)
        for (var i in this.infos)
            this.update(this.infos[i]);
    delete this.infos;
}





function Parameters() {
    this.obj = document.createElement("table");
    this.traits = {};
}
Parameters.prototype.update = function(desc) {
    //Update the parameters descriptor to include the updated values
    for (var name in desc) {
        if (typeof(this.traits[name]) != "undefined" &&
            typeof(desc[name].value) == "undefined") {
            if (this.traits[name].inputs.length > 1) {
                var any = false;
                var tuple = [];
                for (var i = 0; i < this.traits[name].inputs.length; i++) {
                    tuple.push(this.traits[name].inputs[i].value);
                    if (this.traits[name].inputs[i].value)
                        any = true;
                }
                if (any)
                    desc[name].value = tuple;
            } else {
                desc[name].value = this.traits[name].inputs[0].value;
            }
        }
    }
    //clear out the parameters box
    this.obj.innerHTML = "";
    //reinitialize with the updated values
    this.traits = {};
    var func;
    var funcs = {
        "Float":this.add_float,
        "Int": this.add_int,
        "Tuple": this.add_tuple,
        "Array": this.add_array,
        "Instance": this.add_instance,
        "InstanceFromDB": this.add_instance,
        "DataFile": this.add_instance,
        "String":this.add_string,
        "Enum":this.add_enum,
        "OptionsList":this.add_enum,
        "Bool":this.add_bool,
    }


    this.hidden_inputs = [];
    this.hidden_trait_labels = {};

    for (var name in desc) {
        if (funcs[desc[name]['type']]) // if there is a recognized constructor function for the trait type,
            funcs[desc[name]['type']].bind(this)(name, desc[name]); // call the function 
        else
            console.log(desc[name]['type']);
    }

    // console.log(this.hidden_inputs);
    this.hide_attrs();
    // console.log(this.hidden_trait_labels);

}
Parameters.prototype.show_all_attrs = function() {
    for (var attr_name in this.hidden_trait_labels) {
        var label = this.hidden_trait_labels[attr_name];
        label.style.visibility = 'visible';
    }

    for (var k in this.hidden_inputs) {
        var input = this.hidden_inputs[k];
        input.style.visibility = 'visible';
    }
}

Parameters.prototype.hide_attrs = function() {
    for (var attr_name in this.hidden_trait_labels) {
        var label = this.hidden_trait_labels[attr_name];
        label.style.visibility = 'hidden';
    }

    for (var k in this.hidden_inputs) {
        var input = this.hidden_inputs[k];
        input.style.visibility = 'hidden';
    }
}

Parameters.prototype._add = function(name, desc) {
    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    label.innerHTML = name;
    label.setAttribute("for", "param_"+name);
    td.appendChild(label);

    return trait;
}

/*
Function to add an attribute row and label where the 'visibility' attribute of the label can be toggled
*/
Parameters.prototype._add2 = function(name, desc, hidden) {
    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    label.innerHTML = name;
    label.setAttribute("for", "param_"+name);
    
    td.appendChild(label);

    // label.style.visibility = hidden;
    if (hidden === 'hidden') {
        this.hidden_trait_labels[name] = label;
    }

    return trait;
}
Parameters.prototype.add_tuple = function(name, info) {
    var len = info['default'].length;
    var trait = this._add2(name, info['desc'], info['hidden']);
    var wrapper = document.createElement("td");
    wrapper.style.webkitColumnCount = len < 4? len : 4;
    wrapper.style.mozColumnCount = len < 4? len : 4;
    wrapper.style.columnCount = len < 4? len : 4;

    wrapper.style.visibility = info['hidden'];
    trait.appendChild(wrapper);
    this.obj.appendChild(trait);

    this.traits[name] = {"obj":trait, "inputs":[]};

    // Create an input text field for element of the attribute tuple
    for (var i=0; i < len; i++) {
        var input = document.createElement("input");
        input.type = "text";
        input.name = name;
        input.placeholder = JSON.stringify(info['default'][i]);
        if (typeof(info['value']) != "undefined")
            if (typeof(info['value'][i]) != "string")
                input.value = JSON.stringify(info['value'][i]);
            else
                input.value = info['value'][i];
        // input.style.visibility = info['hidden'];
        if (info['hidden'] === 'hidden') {
            this.hidden_inputs.push(input);
        }

        wrapper.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name].inputs[0].id = "param_"+name;
    for (var i in this.traits[name].inputs) {
        var inputs = this.traits[name].inputs
        this.traits[name].inputs[i].onchange = function() {
            if (this.value.length > 0) {
                for (var j in inputs)
                    inputs[j].required = "required";
            } else {
                for (var j in inputs)
                    inputs[j].removeAttribute("required");
            }
        }
    }
}

Parameters.prototype.add_int = function (name, info) {
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An integer value"
    if (typeof(info['value']) != "undefined")
        input.value = info['value'];
    else
        input.value = info['default'];
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_float = function (name, info) {
    //console.log(info)
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A floating point value";
    input.pattern = "-?[0-9]*\.?[0-9]*";
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info.value;
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info.value);
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_bool = function (name, info) {
    //console.log(info)
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "checkbox";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A boolean value"
    if (typeof(info['value']) != "undefined")
        input.checked=info['value'];
    else
        input.checked = info['default'];
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_array = function (name, info) {
    if (info['default'].length < 4) {
        this.add_tuple(name, info);
        for (var i=0; i < this.traits[name].inputs.length; i++)
            this.traits[name].inputs[i].pattern = '[0-9\\(\\)\\[\\]\\.\\,\\s\\-]*';
    } else {
        var trait = this._add2(name, info['desc'], info['hidden']);
        var div = document.createElement("td");
        var input = document.createElement("input");
        // input.style.visibility = info['hidden'];
        if (info['hidden'] === 'hidden') {
            this.hidden_inputs.push(input);
        }
        trait.appendChild(div);
        div.appendChild(input);
        this.obj.appendChild(trait);

        input.type = "text";
        input.name = name;
        input.id = "param_"+name;
        input.title = "An array value";
        input.placeholder = info['default'];
        if (typeof(info['value']) == "string")
            input.value = info['value'];
        else if (typeof(info['value']) != "undefined")
            input.value = JSON.stringify(info['value']);
        input.pattern = /[0-9\(\)\[\]\.\,\s\-]*/;
        this.traits[name] = {"obj":trait, "inputs":[input]};
    }
}
Parameters.prototype.add_string = function (name, info) {
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.placeholder = info['default'];
    if (typeof(info['value']) != "undefined") {
        input.setAttribute("value", info['value']);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_instance = function(name, info) {
    //console.log(info)
    var options = info['options'];
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("select");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i][0];
        opt.innerHTML = options[i][1];
        if ((typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_enum = function(name, info) {
    //console.log(info)
    var options = info['options'];
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("select");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i];
        opt.innerHTML = options[i];
        if ((typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.to_json = function() {
    var jsdata = {};

    // special checkbox handler for arm_visible--to be removed!
    if (typeof(this.traits["arm_visible"]) != "undefined")
        this.traits["arm_visible"].inputs[0].value = this.traits["arm_visible"].inputs[0].checked;


    for (var name in this.traits) {
        if (this.traits[name].inputs.length > 1) {
            var plist = [];
            for (var i = 0; i < this.traits[name].inputs.length; i++) {
                plist.push(this.traits[name].inputs[i].value)
            }
            if (plist[0].length > 0)
                jsdata[name] = plist;
        } else if (this.traits[name].inputs[0].value.length > 0) {
            jsdata[name] = this.traits[name].inputs[0].value;
        }
    }
    return jsdata;
}










//
// TaskInterface class
//
function TaskInterfaceConstructor() {
	var state = "";
	var lastentry = null;

    this.trigger = function(info) {
		if (this != lastentry) {
			if (lastentry) {
				$(window).unload(); // direct away from the page. This stops testing runs, just in case.. TODO not sure if this works with no arguments
				lastentry.tr.removeClass("rowactive active");

                // TODO related to clicking a different task entry than the one already highlighted?
				lastentry.destroy();
			}
			states[this.status].bind(this)(info);
			lastentry = this;
			state = this.status;
		}
        var transitions = triggers[state];
		for (var next in transitions) {
			if (transitions[next].bind(this)(info)) {
				states[next].bind(this)(info);
				this.status = next;
				state = next;
				return;
			}
		}
	};

	var triggers = {
		"completed": {
			stopped: function(info) { return this.idx == null; },
			error: function(info) { return info.status == "error"; }
		},
		"stopped": {
			running: function(info) { return info.status == "running"; },
			testing: function(info) {return info.status == "testing"; }, 
			errtest: function(info) { return info.status == "error"; }
		},
		"running": {
			error: function(info) { return info.status == "error"; },
			completed: function(info) { return info.State == "stopped" || info.status == "stopped"; },
		},
		"testing": {
			errtest: function(info) { return info.status == "error"; },
			stopped: function(info) { return info.State == "stopped" || info.status == "stopped"; },
		},
		"error": {
			running: function(info) { return info.status == "running"; },
			testing: function(info) {return info.status == "testing"; }
		},
		"errtest": {
			running: function(info) { return info.status == "running"; },
			testing: function(info) {return info.status == "testing"; }
		},
	};

    // Functions to run when experiment states are entered
	var states = {
		completed: function() {
			console.log("state = completed");
			$(window).unbind("unload");
			this.tr.addClass("rowactive active");
			$(".active").removeClass("running error testing");
			this.disable();
			$(".startbtn").hide()
			$("#finished_task_buttons").show();
			$("#bmi").hide();
			this.report.deactivate();

			$("#report").show()
			$("#notes").show()		

            // Hack fix. When you select a block from the task interface, force the 'date' column to still be white
            if (this.__date) {
                this.__date.each(function(index, elem) {
                    $(this).css('background-color', '#FFF');
                });
            }

            
            if (this.start_button_pressed) {
                console.log("recorded start button press");
                setTimeout(
                    function () {
                        console.log('callback after pressing stop');
                        te = new TaskEntry(te.idx);
                    },
                    3000
                );
                console.log("finished set timeout");
            }
		},
		stopped: function() {
			console.log("state = stopped")
			$(window).unbind("unload");
			$(".active").removeClass("running error testing");
			this.tr.addClass("rowactive active");
			this.enable();
			$("#stopbtn").hide()
			$("#startbtn").show()
			$("#testbtn").show()
			$("#finished_task_buttons").hide();
			$("#bmi").hide();

			$("#report").show()
			$("#notes").hide()
		},
		running: function(info) {
			console.log("state = running")
			$(window).unbind("unload"); // remove any bindings to 'stop' methods when the page is reloaded (these bindings are added in the 'testing' mode)
			$(".active").removeClass("error testing").addClass("running");
			this.disable();
			$("#stopbtn").show()
			$("#startbtn").hide()
			$("#testbtn").hide()
			$("#finished_task_buttons").hide();
			$("#bmi").hide();
			// this.report.activate();
			$("#report").show()
			$("#notes").show()				
		},
		testing: function(info) {
            console.log("state = testing");
            // if you navigate away from the page during 'test' mode, the 'TaskEntry.stop' function is set to run
			$(window).unload(te.stop);

			$(".active").removeClass("error running").addClass("testing");
			te.disable(); // disable editing of the exp_content interface

			$("#stopbtn").show();

			$("#startbtn").hide()
			$("#testbtn").hide()
			$("#finished_task_buttons").hide()
			$("#bmi").hide();
			// this.report.activate();

			$("#report").show();
			$("#notes").hide();
		},
		error: function(info) {
            console.log("state = error");
			$(window).unbind("unload");
			$(".active").removeClass("running testing").addClass("error");
			this.disable();
			$(".startbtn").hide();
			$("#finished_task_buttons").show();
			$("#bmi").hide();
			this.report.deactivate();

			$("#report").show()
		},
		errtest: function(info) {
            console.log("state = errtest");
            
			$(window).unbind("unload");
			$(".active").removeClass("running testing").addClass("error");
			this.enable();
			$("#stopbtn").hide();
			$("#startbtn").show();
			$("#testbtn").show();
			$("#finished_task_buttons").hide();
			$("#bmi").hide();
			this.report.deactivate();

			$("#report").show()
		}
	};
}


var TaskInterface = new TaskInterfaceConstructor();

//
// TaskEntry constructor
//
function TaskEntry(idx, info){
    /* Constructor for TaskEntry class
     * idx: string of format row\d\d\d where \d\d\d represents the string numbers of the database ID of the block
     */

    // hide the old content
	$("#content").hide(); 

	this.sequence = new Sequence();
	this.params = new Parameters();
	this.report = new Report(TaskInterface.trigger.bind(this));
	
	$("#parameters").append(this.params.obj);
    $("#plots").empty()

    console.log("JS constructing task entry", idx)

	if (idx) { 
        // If the task entry which was clicked has an id (stored in the database)
        // No 'info' is provided--the ID is pulled from the HTML

		// parse the actual integer database ID out of the HTML object name
        if (typeof(idx) == "number") {
            this.idx = idx;
        } else {
            this.idx = parseInt(idx.match(/row(\d+)/)[1]);    
        }
		
		var id_num = this.idx

		// Create a jQuery object to represent the table row
		this.tr = $("#"+idx);
		this.__date = $("#"+idx + " .colDate");
		console.log(this.__date);

		this.status = this.tr.hasClass("running") ? "running" : "completed";
		if (this.status == 'running')
			this.report.activate();

        // Show the wait wheel before sending the request for exp_info. It will be hidden once data is successfully returned and processed (see below)
        $('#wait_wheel').show()

		$.getJSON("ajax/exp_info/"+this.idx+"/", // URL to query for data on this task entry
			{}, // POST data to send to the server
			function (expinfo) { // function to run on successful response
				this.notes = new Notes(this.idx);
                console.log(this)
				this.update(expinfo);
				this.disable();
				$("#content").show("slide", "fast");

                $('#wait_wheel').hide()

				// If the server responds with data, disable reacting to clicks on the current row so that things don't get reset
				this.tr.unbind("click");

                $("#notes textarea").removeAttr("disabled");
			}.bind(this)
			).error(
				function() {
					alert("There was an error accessing task entry " + id_num + ". See terminal for full error message"); 
				}
			);
	} else { 
		// a "new" task entry is being created
		// this code block executes when you click the header of the left table (date, time, etc.)
		this.idx = null;
		console.log('creating task entry null')

		// show the bar at the top left with drop-downs for subject and task
		this.tr = $("#newentry");
        this.tr.show();  // declared in list.html
		this.status = "stopped";

		// Set 'change' bindings to re-run the _task_query function if the selected task or the features change
		$("#tasks").change(this._task_query.bind(this));
		$("#features input").change(this._task_query.bind(this));

		if (info) { // if info is present and the id is null, then this block is being copied from a previous block
			console.log('creating a new JS TaskEntry by copy')
			this.update(info);
			this.enable();
			$("#content").show("slide", "fast");
		} else { // no id and no info suggests that the table header was clicked to create a new block
            console.log('creating a brand-new JS TaskEntry')
            clear_features()
			TaskInterface.trigger.bind(this)({state:''});

            // query the server for information about the task (which generators can be used, which parameters can be set, etc.)
			this._task_query(
                function() {
    				this.enable();
    				$("#content").show("slide", "fast");
    			}.bind(this)
            );
		}
        // make the notes blank and editable
		$("#notes textarea").val("").removeAttr("disabled");

		// Disable reacting to clicks on the current row so that the interface doesn't get reset
		this.tr.unbind("click");
        $('te_table_header').unbind("click");
	}
	
}

function update_available_generators(gens) {
	console.log(gens)
	if (Object.keys(gens).length > 0) {
		$('#seqgen').empty();

		console.log('Updating generator list')
		$.each(gens, function(key, value) {
		    $('#seqgen')
		    .append($('<option>', { value : key })
		    .text(value)); 
		});
	}
}

function clear_features() {
    $("#features input[type=checkbox]").each(
        function() {
            this.checked = false;
       }
    );
}

/* Populate the 'exp_content' template with data from the 'info' object
 */ 
TaskEntry.prototype.update = function(info) {
	console.log("TaskEntry.prototype.update starting");

    // populate the list of generators
	if (Object.keys(info.generators).length > 0) {
		console.log('limiting generators')
		update_available_generators(info.generators);
	} else {
		console.log('not limiting generators!')
	}

    // Update all the sub-parts of the exp_content template separately
	this.sequence.update(info.sequence);
	this.params.update(info.params);
	this.report.update(info.report);
	if (this.notes)
		this.notes.update(info.notes);
	else
		$("#notes").attr("value", info.notes);

	// set the checkboxes for the "visible" and "flagged for backup"
	$('#hidebtn').attr('checked', info.visible);
	$('#backupbtn').attr('checked', info.flagged_for_backup);

	this.expinfo = info;

	// set the 'tasks' drop-down menu to match the 'info'
	$("#tasks option").each(function() {
		if (this.value == info.task)
			this.selected = true;
	})
	// set the 'subjects' drop-down menu to match the 'info'
	$("#subjects option").each(function() {
		if (this.value == info.subject)
			this.selected = true;
	});
	// set checkmarks for all the features specified in the 'info'
	$("#features input[type=checkbox]").each(
        function() {
    		this.checked = false;
    		for (var idx in info.feats) {
    			if (this.name == info.feats[idx])
    				this.checked = true;
    		}
	   }
    );

	
	// List out the data files in the 'filelist'
	// see TaskEntry.to_json in models.py to see how the file list is generated
    var numfiles = 0;
    this.filelist = document.createElement("ul");

	for (var sys in info.datafiles) {
		if (sys == "sequence") { 
			// Do nothing. No point in showing the sequence..
		} else {  
			// info.datafiles[sys] is an array of files for that system
			for (var i = 0; i < info.datafiles[sys].length; i++) {
				// Create a list element to hold the file name
				var file = document.createElement("li");
				file.textContent = info.datafiles[sys][i];
				this.filelist.appendChild(file);
				numfiles++;
			}
		}
	}

	if (numfiles > 0) {
		// Append the files onto the #files field
		$("#files").append(this.filelist).show();

		// make the BMI show up if there's a neural data file linked
		var neural_data_found = false;
		for (var sys in info.datafiles)
			if ((sys == "plexon") || (sys == "blackrock") || (sys == "tdt")) {
				neural_data_found = true;
				break;
			}

		if (neural_data_found){
            // Create the JS object to represent the BMI menu
            this.bmi = new BMI(this.idx, info.bmi, info.notes);
        }
	}

	if (info.sequence) {
		$("#sequence").show()
	} else {
		$("#sequence").hide()
	}

    console.log("TaskEntry.prototype.update done!");
}

TaskEntry.prototype.toggle_visible = function() {
    var btn = $('#hidebtn');
    if (btn.attr('checked') == 'checked') {
        // uncheck the box
        btn.attr('checked', false);

        // send the data
        $.get("/ajax/hide_entry/"+this.idx, 
            {}, 
            function() {
                console.log("Hiding task entry " + te.idx);
                $("#row" + te.idx).css('background-color', 'gray');
            }
        );
    } else { // is hidden, and we want to show
        // uncheck the box
        $('#hidebtn').attr('checked', true);

        // send the data
        $.get("/ajax/show_entry/"+this.idx, 
            {}, 
            function() {
                console.log("Showing task entry " + te.idx);
                $("#row" + te.idx).css('background-color', 'white');
            }
        );
    }
}

TaskEntry.prototype.toggle_backup = function() {
    var btn = $('#backupbtn');
    if (btn.attr('checked') == 'checked') { // is flagged for backup and we want to unflag
        // uncheck the box
        btn.attr('checked', false);

        // send the data
        $.get("/ajax/unbackup_entry/"+this.idx, 
            {}, 
            function() {
                console.log("Unflagging task entry for backup" + te.idx);
            }
        );
    } else { // is hidden, and we want to show
        // uncheck the box
        btn.attr('checked', true);

        // send the data
        $.get("/ajax/backup_entry/"+te.idx,
            {}, 
            function() {
                console.log("Flagging task entry for backup" + te.idx);
            });
    }
}


/* callback for 'Copy Parameters' button
 */
TaskEntry.copy = function() {
    // start with the info saved in the current TaskEntry object
	var info = te.expinfo;

	info.report = {};          // clear the report data
	info.datafiles = {};       // clear the datafile data
	info.notes = "";           // clear the notes
	
	te = new TaskEntry(null, info);
}
/*
 * Destructor for TaskEntry objects
 */
TaskEntry.prototype.destroy = function() {
	$("#content").hide();

    // Destruct the Report object for this TaskEntry
	this.report.destroy();
    
    // Destruct the Sequence object for this TaskEntry 
	this.sequence.destroy();

    // Free the parameters
    if (this.params) {
		$(this.params.obj).remove();
		delete this.params;
	}

    // Remove any designations that this TaskEntry is active/running/errored/etc.
	this.tr.removeClass("rowactive active error");
	$("#content").removeClass("error running testing")

    // Hide the 'files' field
	$("#files").hide();
	$(this.filelist).remove();

	if (this.idx != null) {
		var idx = "row"+this.idx;

        // re-bind a callback to when the row is clicked
		this.tr.click(
            function() {
                te = new TaskEntry(idx);
            }
        )

        // clear the notes field
		this.notes.destroy();

        // clear the BMI
		if (this.bmi !== undefined) {
			this.bmi.destroy();
			delete this.bmi;
		}

	} else {
		//Remove the newentry row
        $('#newentry').hide()

		//Rebind the click action to create a blank TaskEntry form
		this.tr.click(function() {
			te = new TaskEntry(null);
		})

        $('#te_table_header').click(
            function() {
                te = new TaskEntry(null);      
            }
        )
		//Clean up event bindings
		$("#features input").unbind("change");
		$("#tasks").unbind("change");
	}
}

TaskEntry.prototype._task_query = function(callback) {
	console.log('calling TaskEntry.prototype._task_query')
	var taskid = $("#tasks").attr("value");
	var feats = {};
	$("#features input").each(function() { 
		if (this.checked) 
			feats[this.name] = this.checked;	
	});

	$.getJSON("ajax/task_info/"+taskid+"/", feats, 
		function(taskinfo) {
			console.log("Information about task received from the server");
			console.log(taskinfo);

			this.params.update(taskinfo.params);
			if (taskinfo.sequence) {
				$("#sequence").show()
				this.sequence.update(taskinfo.sequence);
			} else
				$("#sequence").hide()

			if (typeof(callback) == "function")
				callback();

			console.log("\tgenerator data");
			console.log("\t", taskinfo.generators);
			if (taskinfo.generators) {
				console.log('taskinfo.generators')
				console.log(taskinfo.generators)
				update_available_generators(taskinfo.generators);
			}
		}.bind(this)
	);
}

TaskEntry.prototype.stop = function() {
    /* Callback for the "Stop Experiment" button
    */
	var csrf = {};
	csrf['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
	$.post("stop", csrf, TaskInterface.trigger.bind(this));
}

/* Callback for the 'Test' button
 */
TaskEntry.prototype.test = function() {
    this.disable();
    return this.run(false); 
}

/* Callback for the 'Start experiment' button
 */
TaskEntry.prototype.start = function() {
	this.disable();
	return this.run(true);
}

TaskEntry.prototype.run = function(save) {
    // activate the report; start listening to the websocket and update the 'report' field when new data is received
    if (this.report){
        this.report.destroy();
    }
    this.report = new Report(TaskInterface.trigger.bind(this)); // TaskInterface.trigger is a function. 
	this.report.activate();

	var form = {};
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(this.get_data());

    // post to different URL depending on whether the data should be saved or not
	var post_url = save ? "/start" : "/test";
	$.post(post_url, form, 
        function(info) {
    		TaskInterface.trigger.bind(this)(info);
    		this.report.update(info);
    		if (info.status == "running") {
                this.new_row(info);
                this.start_button_pressed = true;
            }
	   }.bind(this)
    );
	return false;
}

TaskEntry.prototype.new_row = function(info) { 
	//
    // Add a row to the left hand side table. This function is run after you start a new task using the 'Start Experiment' button
    // 
    console.log('TaskEntry.prototype.new_row: ', info.idx);
    
	this.idx = info.idx;
	this.tr.removeClass("running active error testing")

	// make the row hidden (becomes visible when the start or test buttons are pushed)
	this.tr.hide();
	this.tr.click(function() {
		te = new TaskEntry(null);
	})
	//Clean up event bindings
	$("#features input").unbind("change");
	$("#tasks").unbind("change");

	this.tr = $(document.createElement("tr"));

	// add an id number to the row
	this.tr.attr("id", "row"+info.idx);

	// Write the HTML for the table row
	this.tr.html("<td class='colDate'>Today</td>" + 
				"<td class='colTime' >--</td>" + 
				"<td class='colID'   >"+info.idx+"</td>" + 
				"<td class='colSubj' >"+info.subj+"</td>" + 
				"<td class='colTask' >"+info.task+"</td>");

	
	// Insert the new row after the top row of the table
	$("#newentry").after(this.tr);
	this.tr.addClass("active rowactive running");
	this.notes = new Notes(this.idx);
}

TaskEntry.prototype.get_data = function() {
	var data = {};
	data['subject'] = parseInt($("#subjects").attr("value"));
	data['task'] = parseInt($("#tasks").attr("value"));
	data['feats'] = {};
	$("#experiment #features input").each(function() {
		if (this.checked)
			data.feats[this.value] = this.name;
	})
	data['params'] = this.params.to_json();
	data['sequence'] = this.sequence.get_data();

	return data
}
TaskEntry.prototype.enable = function() {
	$("#parameters input, #features input").removeAttr("disabled");
	if (this.sequence)
		this.sequence.enable();
	if (!this.idx)
		$("#subjects, #tasks").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
	$("#parameters input, #features input").attr("disabled", "disabled");
	if (this.sequence)
		this.sequence.disable();
	if (!this.idx)
		$("#subjects, #tasks").attr("disabled", "disabled");
}

//
// Notes class
//
function Notes(idx) {
	this.last_TO = null;
	this.idx = idx;
	this.activate();
}
Notes.prototype.update = function(notes) {
	//console.log("Updating notes to \""+notes+"\"");
	$("#notes textarea").attr("value", notes);
}
Notes.prototype.activate = function() {
	this._handle_keydown = function() {
		if (this.last_TO != null)
			clearTimeout(this.last_TO);
		this.last_TO = setTimeout(this.save.bind(this), 2000);
	}.bind(this);
	$("#notes textarea").keydown(this._handle_keydown);
}
Notes.prototype.destroy = function() {
	$("#notes textarea").unbind("keydown", this._handle_keydown);
	$("#notes").val("");
	if (this.last_TO != null)
		clearTimeout(this.last_TO);
	this.save();
}
Notes.prototype.save = function() {
	this.last_TO = null;
	$.post("ajax/save_notes/"+this.idx+"/", {
		"notes":$("#notes textarea").attr("value"), 
		'csrfmiddlewaretoken':$("#experiment input[name=csrfmiddlewaretoken]").attr("value")
	});
}







//////////////////// BMI ////////////////////////
/////////////////////////////////////////////////
/*
Runs the code for any button pushing on the BMI training sub-GUI
*/
var goodcells = /^\s*(?:good|bmi)?\s?(?:unit|cell)s?\s?[:-]\s*\n*(.*)$/gim;
var cellnames = /(\d{1,3})\s?(\w{1})/gim;
var parsecell = /(\d{1,3})\s?(\w{1})/;
var parsetime = /^(?:(\d{0,2}):)?(\d{1,2}):(\d{0,2}\.?\d*)$/;

function BMI(idx, info, notes) {
    this.idx = idx;
    try {
        this.cells = goodcells.exec(notes)[1].match(cellnames);
    } catch (e) {
        this.cells = [];
    }

    // Clear the 'Available' and 'Selected' cells fields of any stale options/selections
    $('#cells').empty()
    $('#available').empty()

    this.info = info;
    this.neuralinfo = info['_neuralinfo'];
    delete info['_neuralinfo']

    this.available = {};
    this.selected = {};

    if (this.neuralinfo !== null) {
        if (this.neuralinfo.is_seed) {
            for (var i = 0; i < this.neuralinfo.units.length; i++) {
                var name = this.neuralinfo.units[i][0];
                name += String.fromCharCode(this.neuralinfo.units[i][1]+96);
                this.remove(name);
            }

            this._bindui();
            this.cancel();
        }
    }
}
BMI.zeroPad = function(number, width) {
    width -= number.toString().length;
    if ( width > 0 ) {
        var w = /\./.test(number) ? 2 : 1;
        return new Array(width + w).join('0') + number;
    }
    return number + "";
}
BMI.hms = function(sec) {
    var h = Math.floor(sec / 60 / 60);
    var m = Math.floor(sec / 60 % 60);
    var s = Math.round(sec % 60);
    if (h > 0)
        return h+':'+BMI.zeroPad(m,2)+':'+BMI.zeroPad(s,2);
    return m+':'+BMI.zeroPad(s,2);
}
BMI.ptime = function(t) {
    var chunks = parsetime.exec(t);
    try {
        var secs = parseInt(chunks[2]) * 60;
        secs += parseFloat(chunks[3]);
        if (chunks[1] !== undefined)
            secs += parseInt(chunks[1]) * 60 * 60;

        return secs;
    } catch (e) {
        return null;
    }
}

/**
 * Move an item from one select field 'src' to another select field 'dst'
 * @param {string} name The name of the item
 * @param {object} src The associative array/object the item 'name' is currently in
 * @param {object} dst The associative array/object the item 'name' should be in
 * @param {string} dstname Name of the destination 'option' field
*/
BMI.swap = function(name, src, dst, dstname) {
    if (dst[name] === undefined) {
        var obj;
        if (src[name] !== undefined) {
            obj = src[name].remove()
            delete src[name];
        } else {
            // the option is not in the source menu, so make a new one
            obj = $("<option>"+name+"</option>");
        }

        var names = [name];
        for (var n in dst)
            names.push(n)
        dst[name] = obj;

        if (names.length == 1)
            return $('#'+dstname).append(obj);

        //Rudimentary lexical sort
        names.sort(function(b, a) {
            var an = parsecell.exec(a);
            var bn = parsecell.exec(b);
            var chan = parseInt(bn[1], 10) - parseInt(an[1], 10);
            var chr = (bn[2].charCodeAt(0) - an[2].charCodeAt(0));
            return chan + chr / 10;
        });
        var idx = names.indexOf(name);
        if (idx == 0)
            dst[names[idx+1]].before(obj);
        else
            dst[names[idx-1]].after(obj);
    }
}
BMI.prototype.add = function(name) {
    BMI.swap(name, this.available, this.selected, 'cells');
}
BMI.prototype.remove = function(name) {
    BMI.swap(name, this.selected, this.available, 'available');
}
BMI.prototype.parse = function(cells, addAvail) {
    var names = cells.match(cellnames);
    if (addAvail) {
        for (var i = 0, il = names.length; i < il; i++) {
            this.remove(names[i]);
        }
    } else {
        $("#cells option").each(function(idx, obj) {
            this.remove($(obj).text());
        }.bind(this));

        for (var i = 0, il = names.length; i < il; i++) {
            this.add(names[i]);
        }
        this.update();
    }
}
BMI.prototype.update = function(names) {
    var names = [];
    $('#cells option').each(function(idx, val) {
        names.push($(val).text());
    })
    $('#cellnames').val(names.join(', '));
}

BMI.prototype.set = function(name) {
    var info = this.info[name];
    $("#cells option").each(function(idx, obj) {
        this.remove($(obj).text());
    }.bind(this));

    for (var i = 0; i < info.units.length; i++) {
        var unit = info.units[i];
        var n = unit[0] + String.fromCharCode(unit[1]+96);
        this.add(n);
    }
    this.update();

    $("#bmibinlen").val(info.binlen);
    $("#tstart").val(BMI.hms(info.tslice[0]));
    $("#tend").val(BMI.hms(info.tslice[1]));
    $("#tslider").slider("values", info.tslice);
    $("#bmiclass option").each(function(idx, obj) {
        if ($(obj).text() == info.cls)
            $(obj).attr("selected", "selected");
    });
    $("#bmiextractor option:first").attr("selected", "selected");
}
BMI.prototype.new = function() {
    $("#cells option").each(function(idx, obj) {
        this.remove($(obj).text());
    }.bind(this));
    $("#bmibinlen").val("0.1");
    $("#bminame").replaceWith("<input id='bminame'>");
    //var selected_bmi_class = $("#bmiclass");    
    
    var selected_bmi_class = document.getElementById("bmiclass");
    var strSel = '_'.concat(selected_bmi_class.options[selected_bmi_class.selectedIndex].text);
    
    var new_bmi_name = this.neuralinfo.name.concat(strSel);
    $("#bminame").val(new_bmi_name);
    for (var i = 0; i < this.cells.length; i++) 
        this.add(this.cells[i]);
    $(".bmibtn").show();
    $("#bmi input,select,textarea").attr("disabled", null);
    this.update();
}
BMI.prototype.cancel = function() {
    $("#bmi input,select,textarea").attr("disabled", "disabled");
    $("#bminame").replaceWith("<select id='bminame' />");
    var i = 0;
    for (var name in this.info) {
        $("#bminame").append('<option>'+name+'</option>');
        i++;
    }

    if (i < 1)
        return this.new();

    $(".bmibtn").hide();
    $("#bminame").append("<option value='new'>Create New</option>");
    this.set($("#bminame option:first").text());

    var _this = this;
    $("#bminame").change(function(e){
        if (this.value == 'new')
            _this.new();
        else
            _this.set(this.value);
    })
}
BMI.prototype._bindui = function() {
    $("#tslider").slider({
        range:true, min:0, max:this.neuralinfo.length, values:[0, this.neuralinfo.length],
        slide: function(event, ui) {
            $("#tstart").val(BMI.hms(ui.values[0]));
            $("#tend").val(BMI.hms(ui.values[1]));
        },
    });
    $("#tstart").val(BMI.hms(0));
    $("#tend").val(BMI.hms(this.neuralinfo.length));
    $("#tstart").keyup(function(e) {
        var values = $("#tslider").slider("values");
        var sec = BMI.ptime(this.value);
        if (sec !== null) {
            $("#tslider").slider("values", [sec, values[1]]);
        }
        if (e.which == 13)
            this.value = BMI.hms(sec);
    });
    $("#tend").keyup(function(e) {
        var values = $("#tslider").slider("values");
        var sec = BMI.ptime(this.value);
        if (sec !== null) {
            $("#tslider").slider("values", [values[0], sec]);
        }
        if (e.which == 13)
            this.value = BMI.hms(sec);
    });
    $("#tstart").blur(function() {
        var values = $("#tslider").slider("values");
        this.value = BMI.hms(values[0]);
    });
    $("#tend").blur(function() {
        var values = $("#tslider").slider("values");
        this.value = BMI.hms(values[1]);
    });

    $('#makecell').click(function() {
        var units = $('#available option:selected');
        units.each(function(idx, obj) {
            this.add($(obj).text());
        }.bind(this));
        this.update();
    }.bind(this));

    $('#makeavail').click(function() {
        var units = $('#cells option:selected');
        units.each(function(idx, obj) {
            this.remove($(obj).text());
        }.bind(this));
        this.update();
    }.bind(this));

    $("#cellnames").blur(function(e) {
        console.log($("#cellnames").val())
        this.parse($("#cellnames").val());
    }.bind(this));

    $("#bmitrain").click(this.train.bind(this));
    $("#bmicancel").click(this.cancel.bind(this));
    $("#bmi").show();
}

// Destructor for the BMI sub-menu
BMI.prototype.destroy = function() {
    if (this.neuralinfo !== null) {
        if (this.neuralinfo.is_seed) {
            $("#tslider").slider("destroy");
            $("#tstart").unbind("keyup");
            $("#tstart").unbind("blur");
            $("#tend").unbind("keyup");
            $("#tend").unbind("blur");
            $("#makecell").unbind("click");
            $("#makeavail").unbind("click");
            $("#cellnames").unbind("click");
            $("#cellnames").unbind("blur");
            $("#bmitrain").unbind("click");
            $("#bmicancel").unbind("click");
            $("#bmi").hide();
        }
    }
}

BMI.prototype.train = function() {
    this.update();
    var csrf = $("#experiment input[name=csrfmiddlewaretoken]");
    var data = {};
    data.bminame = $("#bminame").val();
    data.bmiclass = $("#bmiclass").val();
    data.bmiextractor = $("#bmiextractor").val();
    data.cells = $("#cellnames").val();
    data.channels = $("#channelnames").val();
    data.bmiupdaterate = $("#bmiupdaterate").val();
    data.tslice = $("#tslider").slider("values");
    data.ssm = $("#ssm").val();
    data.pos_key = $("#pos_key").val();
    data.kin_extractor = $("#kin_extractor").val();

    data.csrfmiddlewaretoken = csrf.val();

    $.post("/make_bmi/"+this.idx, data, function(resp) {
        if (resp.status == "success") {
            alert("BMI Training queued");
            this.cancel();
        } else
            alert(resp.msg);
    }.bind(this), "json");
}
