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

Sequence.prototype.seqlist_make_default_list = function() {
    $("#seqlist").replaceWith("<select id='seqlist' name='seq_name'><option value='new'>Create New...</option></select>");
}

Sequence.prototype.seqlist_make_input_field = function(curname) {
    $("#seqlist").replaceWith("<input id='seqlist' name='seq_name' type='text' value='"+curname+"' />");
}

Sequence.prototype.seqlist_type = function() {
    return $('#seqlist').prop('tagName').toLowerCase()
}

/* Update the Sequence fieldset
 * 
 */ 
Sequence.prototype.update = function(info) {
    if (info === undefined) {
        return;
    }
    // example info: {3: { name: "gen_fn1:[n_targets=]", state: "saved", static: false}}
    // number of objects probably depends on whether exp_info or task_info server fn is called
    $("#seqlist").unbind("change");

    // remove all the existing options
    for (var id in this.options)
        $(this.options[id]).remove()

    // make sure seqlist is a 'select'
    this.seqlist_make_default_list();
    
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

    if (Object.keys(info).length > 0) {
        $("#seqgen").val(info[id].generator[0]);
        $("#seqlist").val(id);

        this.params.update(info[id].params);
        $("#seqstatic").attr("checked", info[id].static);

        //Bind the sequence list updating function
        var seq_obj = this;
        this._handle_chlist = function () {
            var id = this.value; // 'this' is bound to the options list when it's used inside the callback below
            if (id == "new") {
                seq_obj.edit()
            }
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
        $("#seqgen").change(this._handle_chgen);

        $("#seqstatic,#seqparams input, #seqgen").attr("disabled", "disabled");
    } else {
        this.edit();
        $("#seqgen").change();
    }
}

Sequence.prototype.destroy_parameters = function() {
    if (this.params) {
        $(this.params.obj).remove();  // remove the HTML table with the parameters in it
        delete this.params; // delete the JS object
    }    
}

Sequence.prototype.destroy = function() {
    // clear out the 'options' dictionary
    for (var id in this.options)
        $(this.options[id]).remove()

    this.destroy_parameters();
    $("#seqlist").unbind("change");
    $("#seqgen").unbind("change");
    
    this.seqlist_make_default_list();
}

Sequence.prototype._make_name = function() {
    // Make name for generator
    var gen = $("#sequence #seqgen option").filter(":selected").text()
    var txt = [];
    var d = new Date();
    var datestr =  d.getFullYear()+"."+(d.getMonth()+1)+"."+d.getDate()+" ";

    $("#sequence #seqparams input").each(
        function() {
            txt.push(this.name+"="+this.value);
        }
    )
    return gen+":["+txt.join(", ")+"]-" + datestr
}

Sequence.prototype.edit = function() {
    var _this = this;
    var curname = this._make_name();
    this.seqlist_make_input_field(curname);
    
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
    // only enable the drop-down of the old sequences
    $("#seqlist").removeAttr("disabled");
}
Sequence.prototype.disable = function() {
    // disable the list of old sequences, the parameter inputs, the generator drop-down, the static checkbox
    $("#seqlist, #seqparams input, #seqgen, #seqstatic").attr("disabled", "disabled");
    $('#show_params').attr("disabled", false);
}


Sequence.prototype.get_data = function() {
    // Get data describing the sequence/generator parameters, to be POSTed to the webserver
    if (this.seqlist_type() == "input") {
        // send data about the generator params to the server and create a new sequence
        var data = {};
        data['name']        = $("#seqlist").val();
        data['generator']   = $("#seqgen").val();
        data['params']      = this.params.to_json();
        data['static']      = $("#seqstatic").prop("checked");
        return data;
    } else {
        // running an old sequence, so just send the database ID
        return parseInt($("#sequence #seqlist").val()); 
    }
}

Sequence.prototype.update_available_generators = function(gens) {
    // example gens: { 1: "gen_fn1", 2: "gen_fn2" }
    if (Object.keys(gens).length > 0) {
        $('#seqgen').empty();
        $.each(gens, function(key, value) {
            $('#seqgen')
            .append($('<option>', { value : key })
            .text(value)); 
        });
    }
}