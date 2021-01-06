// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(`<fieldset id="sequence">
        <div id="seqparams" hidden="true">
            Parameters
        </div>

        <select id="seqlist" name="seq_name">
            <option value="new">Create New...</option>
        </select>

        <select id="seqgen" name="seq_gen"  hidden="true">
            <option value="gen1">gen1</option>
            <option value="gen2">gen2</option>
        </select>

        <input id="seqstatic" type="checkbox" name="seqstatic">
        <label for="seqstatic">Static</label>
    </fieldset>`);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}

function Sequence() {
    // create a new Parameters object
    var params = new Parameters();
    this.params = params

    // add the empty HTML table to the Sequence field (parameters will be populated into rows of the table once they are known, later)
    $("#seqparams").append(this.params.obj);

    var _this = this;
    _handle_set_new_name = function() { 
        $("#seqlist option[value=\"new\"]").attr("name", _this._make_name());
    };
    this._handle_set_new_name = _handle_set_new_name
    _handle_chgen = function() {
        $.getJSON("/ajax/gen_info/"+this.value+"/",
            {},
            function(info) {
                params.update(info.params);
                _handle_set_new_name();
                $("#seqparams input").change(_handle_set_new_name);
            }
        );
    }
    $("#seqgen").change(_handle_chgen);

    $("#seqparams").click(
        // if you click on the parameters table and the drop-down list of
        // available sequences (not generators) is enabled, then enable editing
        function() {
            if ($("#seqparams").attr("disabled") == "disabled")
                this.edit();
        }.bind(this)
    );
    this.options = {};
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
    $("#seqlist").replaceWith(
        "<select id='seqlist' name='seq_name'><option value='new'>Create New...</option></select>"
    );

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
                $("#seqstatic,#seqparams,#seqparams input, #seqgen").attr("disabled", "disabled");

                // change the value of the generator drop-down list to the generator for this sequence.
                $('#seqgen').val(info[id].generator[0]);

                // mark the static checkbox, if the sequence was static
                $("#seqstatic").attr("checked", info[id].static);
            }
        };
        $("#seqlist").change(this._handle_chlist);
        $("#seqstatic,#seqparams,#seqparams input, #seqgen").attr("disabled", "disabled");
    } else {
        this.edit();
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
}

Sequence.prototype._make_name = function() {
    // Make name for generator
    var gen = $("#sequence #seqgen option").filter(":selected").text()
    var txt = [];
    var d = new Date();
    // var datestr =  d.getFullYear()+"."+(d.getMonth()+1)+"."+d.getDate()+" ";

    $("#sequence #seqparams input").each(
        function() {
            if (this.value)
                txt.push(this.name+"="+this.value);
            else
                txt.push(this.name+"="+this.placeholder);
        }
    )
    // return gen+":["+txt.join(", ")+"]-" + datestr
    return gen+":["+txt.join(", ")+"]"
}

Sequence.prototype.edit = function() {
    $("#seqlist").val("new");
    $("#seqparams input").val("");
    this._handle_set_new_name();
    $("#seqgen, #seqparams, #seqparams input, #seqstatic").removeAttr("disabled");
    $("#seqgen").change();
}


Sequence.prototype.enable = function() {
    // only enable the drop-down of the old sequences
    $("#seqlist").removeAttr("disabled");
}
Sequence.prototype.disable = function() {
    // disable the list of old sequences, the parameter inputs, the generator drop-down, the static checkbox
    $("#seqlist, #seqparams, #seqparams input, #seqgen, #seqstatic").attr("disabled", "disabled");
    $('#show_params').attr("disabled", false);
}


Sequence.prototype.get_data = function() {
    // Get data describing the sequence/generator parameters, to be POSTed to the webserver
    var val = $("#seqlist").val();
    var name = $("#seqlist option[value=\"new\"]").attr("name");
    var id = null

    // Try to match the name to a previous sequence
    $("#seqlist option").each(function(){
        if ($(this).html() == name) id = parseInt($(this).val());
    });
    
    if (val == "new" && id != null) {
        // old sequence exists, just a duplicate of a previous ID
        return id;
    } else if (val == "new") {
        // send data about the generator params to the server and create a new sequence
        var data = {};
        data['name']        = name;
        data['generator']   = $("#seqgen").val();
        data['params']      = this.params.to_json();
        data['static']      = $("#seqstatic").prop("checked");
        return data;
    } else {
        // running an old sequence, so just send the database ID
        return parseInt(val);
    }
}

Sequence.prototype.update_available_generators = function(gens) {
    if (this.gens && JSON.stringify(gens)==JSON.stringify(this.gens)) return; // don't update if it's the same as before
    this.gens = gens
    // example gens: [[1, "gen_fn1"], [2, "gen_fn2"]]
    if (gens.length > 0) {
        $('#seqgen').empty(); // TODO no need to destroy all of them since they're already populated. just filter out the unused ones
        for (var i=0; i<gens.length; i++) {
            $('#seqgen')
            .append($('<option>', { value : gens[i][0]})
            .text(gens[i][1]));
        }
    }
}

if (typeof(module) !== 'undefined' && module.exports) {
  exports.Sequence = Sequence;
  exports.$ = $;
}
