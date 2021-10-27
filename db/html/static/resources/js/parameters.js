// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(``);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}

function Parameters(editable=false) {
    this.obj = document.createElement("table");
    this.traits = {};
    this.editable = editable;
}
Parameters.prototype.update = function(desc) {
    // Update the parameters descriptor to include the updated values
    // "desc" is a JSON object of form {"param1": {"value": value, "type": type, "desc": string description}, "param2": ...}
    // if the parameter is a drop-down, the parameter's info should also have an "options" field
    for (var name in desc) {
        if (typeof(this.traits[name]) != "undefined" && typeof(desc[name].value) == "undefined") {
            var trait = this.traits[name]
            if (trait.inputs.length > 1) { // more than one input entry field, e.g., for tuple traits
                var any = false; // flag indicating that the tuple had any input specified
                var tuple = [];
                for (var i = 0; i < trait.inputs.length; i++) {
                    tuple.push(trait.inputs[i].value);
                    if (trait.inputs[i].value) {
                        any = true;
                    }
                }
                if (any)
                    desc[name].value = tuple;
            } else if (desc[name]['type'] == 'Bool') {
                desc[name].value = trait.inputs[0].checked;
            } else {
                desc[name].value = trait.inputs[0].value;
            }
        }
    }

    // clear out the parameters box
    this.obj.innerHTML = "";
    // reinitialize
    this.traits = {};
    this.hidden_parameters = [];

    // append the new values
    this.append(desc);

    // show everything
    this.show_all_attrs();

}

Parameters.prototype.append = function(desc) {
    // append the given traits to the old traits

    var funcs = {
        "Float" :           this.add_float,
        "Int":              this.add_int,
        "Tuple":            this.add_tuple,
        "Array":            this.add_array,
        "Instance":         this.add_instance,
        "InstanceFromDB":   this.add_instance,
        "DataFile":         this.add_instance,
        "String":           this.add_string,
        "Enum":             this.add_enum,
        "OptionsList":      this.add_enum,
        "Bool":             this.add_bool,
        "List":             this.add_list,
    }

    for (var name in desc) {
        if (funcs[desc[name]['type']]) {// if there is a recognized constructor function for the trait type,
            var fn = funcs[desc[name]['type']].bind(this);
            fn(name, desc[name]); // call the function
        }
        else
            debug(desc[name]['type']);
    }
}

Parameters.prototype.show_all_attrs = function() {


    for (var name in this.hidden_parameters) {
        if ($('#show_params').prop('checked'))
            $(this.hidden_parameters[name]).show();
        else
            $(this.hidden_parameters[name]).hide();
    }

}

Parameters.prototype.enable = function() {
    $(this.obj).find("input, select, checkbox").removeAttr("disabled");
}
Parameters.prototype.disable = function() {
    $(this.obj).find("input, select, checkbox").attr("disabled", "disabled");
}

/*
Function to add an attribute row and label where the 'visibility' attribute of the label can be toggled
*/
Parameters.prototype.add_to_table = function(name, info) {
    let desc = info["desc"];
    let hidden = info["hidden"];
    let label_text = info["label"];

    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    td.style.textAlign = "right";

    if (label_text != undefined) {
        label.innerHTML = label_text;
    } else {
        label.innerHTML = name;
    }

    label.setAttribute("for", "param_"+name);

    td.appendChild(label);

    // optionally add a minus button
    if (this.editable && !info["required"]) {
        var remove_row = document.createElement("input");
        remove_row.setAttribute("class", "paramremove");
        remove_row.setAttribute("type", "button");
        remove_row.setAttribute("value", "-");
        var this_ = this;
        $(remove_row).on("click", function() {this_.remove_row(name);});
        td.appendChild(remove_row);
    }

    // label.style.visibility = hidden;
    if (hidden === 'hidden') {
        this.hidden_parameters[name] = $(label).closest("tr");
    }

    return trait;
}
Parameters.prototype.add_tuple = function(name, info) {
    var len = info['default'].length;
    var trait = this.add_to_table(name, info);
    var wrapper = document.createElement("td");
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
        if (input.value == input.placeholder)
            input.value = null
        wrapper.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name].inputs[0].id = "param_"+name;
    for (var i in this.traits[name].inputs) {
        var inputs = this.traits[name].inputs
        this.traits[name].inputs[i].onchange = function() {
            if (this.value.length > 0) {
                for (var j in inputs)
                    if (inputs[j].placeholder.length == 0) inputs[j].required = "required";
            } else {
                for (var j in inputs)
                    inputs[j].removeAttribute("required");
            }
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
        this.traits[name].inputs[i].onchange();
    }
}

Parameters.prototype.add_int = function (name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    if (typeof(info['value']) != "undefined")
        input.value = info['value'];
    else
        input.value = info['default'];
    if (info['required']) {
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
        input.onchange();
    } 
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_float = function (name, info) {
    //debug(info)
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.pattern = "-?[0-9]*\.?[0-9]*";
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info.value;
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info.value);
    if (input.value == input.placeholder)
        input.value = null;
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
        input.onchange();
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_bool = function (name, info) {
    //debug(info)
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "checkbox";
    input.name = name;
    input.id = "param_"+name;
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
        this.add_list(name, info);
    }
}
Parameters.prototype.add_string = function (name, info) {
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    $(input).addClass("string");
    input.name = name;
    input.id = "param_"+name;
    input.placeholder = info['default'];
    if (typeof(info['value']) != "undefined") {
         input.setAttribute("value", info['value']);
    }
    if (input.value == input.placeholder)
        input.value = null;
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
        input.onchange();
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_instance = function(name, info) {
    //debug(info)
    var options = info['options'];
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("select");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    if (info['required']) {
        var opt = document.createElement("option");
        opt.setAttribute("selected", "selected");
        input.appendChild(opt);
        input.required = "required";
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
    }
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i][0];
        opt.innerHTML = options[i][1];
        if (!info['required'] && 
            (typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_enum = function(name, info) {
    //debug(info)
    var options = info['options'];
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("select");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    if (info['required']) {
        var opt = document.createElement("option");
        opt.setAttribute("selected", "selected");
        input.appendChild(opt);
        input.required = "required";
        input.onchange = function() {
            if (this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
    }
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i];
        opt.innerHTML = options[i];
        if (!info['required'] &&
            (typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_list = function(name, info) { // comma separated list of string values
    var trait = this.add_to_table(name, info);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.is_list = true;
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info['value'];
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info['value']);
    if (input.value == input.placeholder)
        input.value = null
    if (info['required']) {
        input.onchange = function() {
            if (this.placeholder.length == 0 && this.value.length == 0)
                this.required = "required";
            else if (this.required)
                this.removeAttribute("required");
        }
        input.onchange();
    }
    input.pattern = /[0-9\(\)\[\]\.\,\s\-]*/;
    this.traits[name] = {"obj":trait, "inputs":[input], "list":true};
}
/*
* Function to ask user for a new row
*/
Parameters.prototype.add_row = function() {
    var trait = $("<tr>");
    trait.attr("title", "New metadata entry");
    var td = $("<td>");
    td.addClass("param_label");
    td.css("textAlign", "right");
    trait.append(td);
    var label = $('<input type="text" placeholder="New entry">');
    label.css({"border": "none", "border-color": "transparent"});
    var div = $("<td>");
    var input = $('<input type="text" class="string" required>');
    input.on("change", function() {
        if (this.value.length == 0)
            this.required = "required";
        else if (this.required)
            this.removeAttribute("required");
    })
    td.append(label);
    trait.append(div);
    div.append(input);
    $(this.obj).append(trait);

    var _this = this;
    label.blur(function(){
        var name = label.val();
        if (name) {

            // If a label has been set, make it permanent
            trait.attr('id', 'param_'+name);
            new_label = $("<label>")
            new_label.css("textAlign", "right");
            new_label.addClass("string");
            new_label.html(name);
            label.replaceWith(new_label);
            _this.traits[name] = {"obj":trait.get(0), "inputs":[input.get(0)]}
            if (_this.editable) {
                // add a minus button
                var remove_row = document.createElement("input");
                remove_row.setAttribute("class", "paramremove");
                remove_row.setAttribute("type", "button");
                remove_row.setAttribute("value", "-");
                $(remove_row).on("click", function() {_this.remove_row(name);});
                new_label.after(remove_row);
            }
        } else {

            // Otherwise delete the row
            trait.remove();
        }
    })
}

Parameters.prototype.remove_row = function(name) {
    if (typeof(this.traits[name]) != "undefined") {
        var trait = this.traits[name]
        trait.obj.remove();
        delete this.traits[name];
    }
}

function get_param_input(input_obj) {
    if (input_obj.type == 'checkbox') {
        return input_obj.checked;
    } else if (input_obj.is_list && input_obj.value.length > 0) {
        var list = input_obj.value.replace(/\[|\]/g,"").split(/[ ,]+/);
        if (Array.isArray(list)) return list;
        else return [list] // force it to be a list even if one element
    } else if (input_obj.is_list) {
        var list = input_obj.placeholder
        if (Array.isArray(list)) return list;
        else return [list] // force it to be a list even if one element
    } else if (input_obj.value.length > 0) {
        return input_obj.value;
    } else {
        return input_obj.placeholder;
    }
}

Parameters.prototype.to_json = function(get_all) {
    var jsdata = {};

    for (var name in this.traits) {
        var trait = this.traits[name];
        if (trait.inputs.length > 1) { // tuple/array trait
            // put all the input options into a list
            var plist = [];
            for (var i = 0; i < trait.inputs.length; i++) {
                plist.push(get_param_input(trait.inputs[i]))
            }
            jsdata[name] = plist;
        } else {
            jsdata[name] = get_param_input(trait.inputs[0]);
        }
    }
    return jsdata;
}

Parameters.prototype.clear_all = function() {
    for (var name in this.traits) {
        var trait = this.traits[name];
        for (var i = 0; i < trait.inputs.length; i++) {
            trait.inputs[i].value = null;
            if (trait.inputs[i].onchange)
                trait.inputs[i].onchange();
        }
    }
}


if (typeof(module) !== 'undefined' && module.exports) {
  exports.Parameters = Parameters;
  exports.$ = $;
}

