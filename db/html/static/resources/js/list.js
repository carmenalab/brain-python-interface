
var log_mode = 2

function log(msg, level) {
    if (level <= log_mode) {
        console.log(msg)
    }
}

function debug(msg) {
    log(msg, 5);
}

function remove_entries(start, end) { // for debugging
    for (i=start; i<end; i++) $.ajax("ajax/remove_entry/"+i);
}

//
// TaskInterface class
//
function interface_fn_completed() {
    log("state = completed", 2);
    $(window).unbind("unload");
    this.tr.removeClass("running error testing").addClass("rowactive active");
    $("#content").removeClass("running error testing");
    this.disable();
    $("#start_buttons").hide()
    $("#stop_buttons").hide();
    $("#finished_task_buttons").show();
    $("#bmi").hide();
    
    $("#report").show()
    $("#notes").show()      
    this.controls.deactivate();
    this.report.deactivate();
    this.report.set_mode("completed");

    // Hack fix. When you select a block from the task interface, force the 'date' column to still be white
    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }

    // If we just finished running, reload the info from the server
    if (this.start_button_pressed) setTimeout(function(){ te.reload(); }, 1000);
}

function interface_fn_stopped() {
    log("state = stopped", 2)
    $(window).unbind("unload");
    $("#content").removeClass("running error testing");
    this.tr.removeClass("running error testing").addClass("rowactive active");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();

    $("#report").show()
    $("#notes").hide()
    this.controls.deactivate();

    // Hack fix. When you select a block from the task interface, force the 'date' column to still be white
    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }
}

function interface_fn_running(info) {
    log("state = running", 2)
    $(window).unbind("unload"); // remove any bindings to 'stop' methods when the page is reloaded (these bindings are added in the 'testing' mode)
    this.tr.removeClass("error testing").addClass("running");
    $('#content').removeClass("error testing").addClass("running");
    this.disable();
    $("#stop_buttons").show()
    $("#start_buttons").hide();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.activate();

    $("#report").show()
    $("#notes").show()   
    this.controls.activate();   
    
    // Hack fix. When you select a block from the task interface, force the 'date' column to still be white
    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }
}

function interface_fn_testing(info) {
    log("state = testing", 2);
    // if you navigate away from the page during 'test' mode, the 'TaskEntry.stop' function is set to run
    $(window).unload(te.stop);

    this.tr.removeClass("error running").addClass("testing");
    $('#content').removeClass("error running").addClass("testing");
    te.disable(); // disable editing of the exp_content interface

    $("#stop_buttons").show();
    $("#start_buttons").hide()
    $("#finished_task_buttons").hide()
    $("#bmi").hide();
    this.report.activate();

    $("#report").show();
    $("#notes").hide();
    this.controls.activate();
}

function interface_fn_error(info) {
    log("state = error", 2);
    $(window).unbind("unload");
    this.tr.removeClass("running testing").addClass("error");
    $('#content').removeClass("running testing").addClass("error");
    this.disable();
    $("#start_buttons").hide();
    $("#finished_task_buttons").show();
    $("#bmi").hide();
    this.report.deactivate();

    $("#report").show()
    this.controls.deactivate();
}
function interface_fn_errtest(info) {
    log("state = errtest", 2);

    $(window).unbind("unload");
    this.tr.removeClass("running testing").addClass("error");
    $('#content').removeClass("running testing").addClass("error");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.deactivate();

    $("#report").show()
    this.controls.deactivate();
}

function TaskInterfaceConstructor() {
    debug("TaskInterfaceConstructor")
    var state = "";
    var lastentry = null;

    // 'this' is always bound to the active task entry before calling
    // The 'trigger' function is usually called as a callback, either through
    // changes in Report data from the server or button presses
    this.trigger = function(info) {
        debug("TaskInterfaceConstructor.trigger");
        debug(this)
        debug(info)
        if (this != lastentry) {
            debug(2)
            if (lastentry && !lastentry.destroyed) {
                $(window).unload(); // direct away from the page. This stops testing runs, just in case..
                lastentry.destroy(); // remove the previously highlighted entry
            }
            state = this.status;
            states[state].bind(this)(info);
            lastentry = this;
        }

        var transitions = fsm_transition_table[state];
        for (var next_state in transitions) {
            let _test_next_state = transitions[next_state].bind(this);
            if (_test_next_state(info)) {
                debug("executing transition...");
                debug(info)
                let _start_next_state = states[next_state].bind(this);
                _start_next_state(info);
                this.status = next_state;
                state = next_state;
                return;
            }
        }
        debug("No transition found!");
    };

    var fsm_transition_table = {
        "completed": {
            stopped: function(info) { return this.idx == null; },
            running: function(info) { return info.status == "running"; },
            testing: function(info) { return info.status == "testing"; },
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
        "completed": interface_fn_completed,
        "stopped": interface_fn_stopped,
        "running": interface_fn_running,
        "testing": interface_fn_testing,
        "error": interface_fn_error,
        "errtest": interface_fn_errtest,
    };
}


var task_interface = new TaskInterfaceConstructor();

function Files() {
    this.neural_data_found = false;
    $("#file_modal_server_resp").html("");
}
Files.prototype.hide = function() {
    $("#files").hide();
}
Files.prototype.show = function() {
    $("#files").show();
}
Files.prototype.clear = function() {
    $("#file_list").html("")
}
Files.prototype.update_filelist = function(datafiles, task_entry_id) {
    // List out the data files in the 'filelist'
    // see TaskEntry.to_json in models.py to see how the file list is generated
    var numfiles = 0;
    this.filelist = document.createElement("ul");

    for (var sys in datafiles) {
        for (var i = 0; i < datafiles[sys].length; i++) {
            // Create a list element to hold the file name
            var file = document.createElement("li");
            file.textContent = datafiles[sys][i];
            this.filelist.appendChild(file);
            numfiles++;
        }
    }

    if (numfiles > 0) {
        // Append the files onto the #files field
        $("#file_list").append(this.filelist);
        for (var sys in datafiles) {
            if (sys != "hdf") {
                this.neural_data_found = true;
                break;
            }
        }
    }
}

//
// TaskEntry constructor
//
function TaskEntry(idx, info) {
    debug("TaskEntry constructor")
    /* Constructor for TaskEntry class
     * idx: string of format row\d\d\d where \d\d\d represents the string numbers of the database ID of the block
     */

    // resize the window to fit the TE pane correctly
    $(window).resize()

    // hide the old content
    $("#content").hide();

    // Reset HTML fields
    $("#file_list").empty();
    $("#content").removeClass("error running testing")
    $("#files").hide();
    $('#newentry').hide()
    $('#te_table_header').click(
        function() {
            if (te) te.destroy();
            te = new TaskEntry(null);
        }
    )
    $("#tasks").unbind("change");

    // Make new widgets
    this.sequence = new Sequence();
    this.params = new Parameters();
    this.report = new Report(task_interface.trigger.bind(this));
    this.files = new Files();
    this.controls = new Controls();
    this.controls.hide()

    $("#parameters").append(this.params.obj);
    $("#plots").empty()

    debug("JS constructing task entry", idx)

    if (idx) {
        // If the task entry which was clicked has an id (stored in the database)
        // No 'info' is provided--the ID is pulled from the HTML
        this.status = 'completed'

        // parse the actual integer database ID out of the HTML object name
        if (typeof(idx) == "number") {
            this.idx = idx;
            idx = "row" + idx;
        } else {
            this.idx = parseInt(idx.match(/row(\d+)/)[1]);
        }

        var id_num = this.idx

        // Create a jQuery object to represent the table row
        this.tr = $("#"+idx);
        this.__date = $("#"+idx + " .colDate");
        debug(this.__date);

        // Show the wait wheel before sending the request for exp_info. It will be hidden once data is successfully returned and processed (see below)
        $('#wait_wheel').show();
        $('#tr_seqlist').hide();

        $.getJSON("ajax/exp_info/"+this.idx+"/", // URL to query for data on this task entry
            {}, // POST data to send to the server
            function (expinfo) { // function to run on successful response
                this.notes = new Notes(this.idx);
                debug(this)
                this.update(expinfo);
                $("#content").show("slide", "fast");

                $('#wait_wheel').hide()

                // If the server responds with data, disable reacting to clicks on the current row so that things don't get reset
                this.tr.unbind("click");

                // debug('setting ')
                this.tr.addClass("rowactive active");
                $("#newentry").hide();

                // enable editing of the notes field for a previously saved entry
                $("#notes textarea").removeAttr("disabled");
            }.bind(this)
            ).error(
                function() {
                    alert("There was an error accessing task entry " + id_num + ". See terminal for full error message");
                    this.tr.removeClass("rowactive active error");
                    $('#wait_wheel').hide();
                }.bind(this)
            );
    } else {
        // a "new" task entry is being created
        // this code block executes when you click the header of the left table (date, time, etc.)
        this.idx = null;
        $("#entry_name").val("");
        this.status = "stopped";
        this.tr = $("#newentry");
        this.tr.show(); // make sure the task entry row is visible 
        $('#tr_seqlist').show();
        
        if (info) { // if info is present and the id is null, then this block is being copied from a previous block
            debug('creating a new JS TaskEntry by copy')
            info.state = "completed";
            this.update(info);
            
            var taskid = $("#tasks").attr("value");
            var sel_feats = feats.get_checked_features();
            this.files.hide();
        } else { // no id and no info suggests that the table header was clicked to create a new block
            debug('creating a brand-new JS TaskEntry')
            feats.clear();
            this.report.hide();
            this.files.hide();
            // task_interface.trigger.bind(this)({state:''});
        }

        // Set 'change' bindings to re-run the _task_query function if the selected task or the features change
        $("#tasks").change(this._task_query.bind(this));
        feats.bind_change_callback(this._task_query.bind(this))

        // query the server for information about the task (which generators can be used, which parameters can be set, etc.)
        this._task_query(
            function() {
                this.enable();
                $("#content").show("slide", "fast");
            }.bind(this)
        );

        // make the notes blank and editable
        $("#notes textarea").val("").removeAttr("disabled");

        // Disable reacting to clicks on the current row so that the interface doesn't get reset
        this.tr.unbind("click");
        $('te_table_header').unbind("click");

        task_interface.trigger.bind(this)({status: this.status});
    }

    this.being_copied = false;
}
/* Populate the 'exp_content' template with data from the 'info' object
 */
TaskEntry.prototype.update = function(info) {
    debug("TaskEntry.prototype.update");

    // populate the list of generators
    if (Object.keys(info.generators).length > 0) {
        debug('limiting generators')
        this.sequence.update_available_generators(info.generators);
    } else {
        debug('not limiting generators!')
    }

    this.status = info.state;

    // Update all the sub-parts of the exp_content template separately
    this.sequence.update(info.sequence);
    this.params.update(info.params);
    if (this.notes)
        this.notes.update(info.notes);
    else
        $("#notes").attr("value", info.notes);
    feats.unbind_change_callback();
    this.report.update(info.report);
    
    // set the checkboxes for the "visible" and "flagged for backup"
    $('#hidebtn').attr('checked', info.visible);
    $('#backupbtn').attr('checked', info.flagged_for_backup);
    $('#templatebtn').attr('checked', info.template);

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

    feats.select_features(info.feats);

    collections.select_collections(info.collections);

    $("#entry_name").val(info.entry_name);

    this.files.show();
    this.files.update_filelist(info.datafiles, this.idx);

    if (this.files.neural_data_found){
        // Create the JS object to represent the BMI menu
        this.bmi = new BMI(this.idx, info.bmi, info.notes);
    }

    if (info.sequence) {
        $("#sequence").show()
    } else {
        $("#sequence").hide()
    }

    if (this.status != "stopped") this.disable();
    task_interface.trigger.bind(this)({status: this.status});

    debug("TaskEntry.prototype.update done!");
}
TaskEntry.prototype.reload = function() {
    this.files.clear();

    if (this.idx == null) return;

    $.getJSON("ajax/exp_info/"+this.idx+"/", // URL to query for data on this task entry
        {}, // POST data to send to the server
        function (expinfo) { // function to run on successful response
            this.notes.destroy();
            this.notes = new Notes(this.idx);
            this.sequence.destroy();
            this.sequence = new Sequence();
            debug(this)
            this.update(expinfo);
            $("#content").show("slide", "fast");

            $('#wait_wheel').hide()

            // If the server responds with data, disable reacting to clicks on the current row so that things don't get reset
            this.tr.unbind("click");

            // debug('setting ')
            this.tr.addClass("rowactive active");

            // enable editing of the notes field for a previously saved entry
            $("#notes textarea").removeAttr("disabled");
        }.bind(this)
        ).error(
            function() {
                alert("There was an error accessing task entry " + id_num + ". See terminal for full error message");
                this.tr.removeClass("rowactive active error");
                $('#wait_wheel').hide();
            }.bind(this)
        );
}

TaskEntry.prototype.toggle_visible = function() {
    debug("TaskEntry.prototype.toggle_visible")
    var btn = $('#hidebtn');
    if (btn.is(':checked')) { // is hidden, and we want to show
        $.get("/exp_log/ajax/show_entry/"+this.idx,
            {},
            function() {
                debug("Showing task entry " + te.idx);
                $("#row" + te.idx).css({'background-color': 'white'});
            }
        );
    } else { // want to hide
        $.get("/exp_log/ajax/hide_entry/"+this.idx,
            {},
            function() {
                debug("Hiding task entry " + te.idx);
                $("#row" + te.idx).css({"background-color": "gray"});
                te.destroy();
            }
        );
    }
}

TaskEntry.prototype.save_name = function() {
    $.post("save_entry_name", {"id": this.idx, "entry_name": $("#entry_name").val()});
}

TaskEntry.prototype.toggle_backup = function() {
    debug("TaskEntry.prototype.toggle_backup")
    var btn = $('#backupbtn');
    if (btn.is(':checked')) {  // is hidden, and we want to show
        $.get("/exp_log/ajax/backup_entry/"+te.idx,
            {},
            function() {
                debug("Flagging task entry for backup" + te.idx);
        });
    } else {
        $.get("/exp_log/ajax/unbackup_entry/"+this.idx,
            {},
            function() {
                debug("Unflagging task entry for backup" + te.idx);
            }
        );
    }
}

TaskEntry.prototype.toggle_template = function() {
    debug("TaskEntry.prototype.toggle_template")
    var btn = $('#templatebtn');
    if (btn.is(':checked')) {  // is hidden, and we want to show
        $.get("/exp_log/ajax/template_entry/"+this.idx,
            {},
            function() {
                debug("Flagging task entry as a template: " + te.idx);
        });
    } else {
        $.get("/exp_log/ajax/untemplate_entry/"+this.idx,
            {},
            function() {
                debug("Unflagging task entry as a template: " + te.idx);
            }
        );
    }
}


/* callback for 'Copy Parameters' button. Note this is not a prototype function
 */
TaskEntry.copy = function() {
    debug("TaskEntry.copy")
    // start with the info saved in the current TaskEntry object
    var info = te.expinfo;

    if (info != null) {
        info.report = {};          // clear the report data
        info.datafiles = {};       // clear the datafile data
        info.notes = "";           // clear the notes
    }

    te.being_copied = true;
    te.destroy();
    te = new TaskEntry(null, info);
    $('#report').hide();        // creating a TaskEntry with "null" goes into the "stopped" state
}

/*
 * Destructor for TaskEntry objects
 */
TaskEntry.prototype.destroy = function() {
    debug("TaskEntry.prototype.destroy")
    $("#content").hide();

    // Destruct objects
    this.report.destroy();
    if (this.being_copied) {
        // don't destroy when copying because two objects try to manipulate the
        // Sequence at the same time
        this.sequence.destroy_parameters();
    } else {
        this.sequence.destroy();
    }
    if (this.params)  $(this.params.obj).remove();
    if (this.notes) this.notes.destroy();
    if (this.bmi) this.bmi.destroy();
    $(this.filelist).remove();

    // Remove any designations that this TaskEntry is active/running/errored/etc.
    this.tr.removeClass("rowactive active error");

    // Re-bind a callback to when the row is clicked
    var idx = this.idx
    this.tr.click(
        function() {
            if (te) te.destroy();
            te = new TaskEntry(idx);
        }
    );

    this.destroyed = true;
}

TaskEntry.prototype.remove = function(callback) {
    debug('TaskEntry.prototype.remove')
    $.getJSON("ajax/remove_entry/"+this.idx,function() {
        location.reload();
    });
}

TaskEntry.prototype._task_query = function(callback) {
    debug('TaskEntry.prototype._task_query')
    var taskid = $("#tasks").attr("value");
    var sel_feats = feats.get_checked_features();

    $.getJSON("ajax/task_info/"+taskid+"/", sel_feats,
        function(taskinfo) {
            debug("Information about task received from the server");
            debug(taskinfo);

            this.params.update(taskinfo.params);
            if (taskinfo.sequence) {
                $("#sequence").show()
                this.sequence.update(taskinfo.sequence);
            } else
                $("#sequence").hide()

            if (typeof(callback) == "function")
                callback();

            if (taskinfo.generators) {
                this.sequence.update_available_generators(taskinfo.generators);
            }

            if (taskinfo.controls) {
                this.controls.update(taskinfo.controls);
            } else {
                this.controls.update([]);
            }
        }.bind(this)
    );
}

function stop_fn_callback(resp) {
    debug("Stop callback received");
}

TaskEntry.prototype.stop = function() {
    /* Callback for the "Stop Experiment" button
    */
    debug("TaskEntry.prototype.stop")
    var csrf = {};
    csrf['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
    $.post("stop/", csrf, task_interface.trigger.bind(this));
}

/* Callback for the 'Test' button
 */
TaskEntry.prototype.test = function() {
    debug("TaskEntry.prototype.test")
    this.disable();
    return this.run(false, true);
}

/* Callback for the 'Start experiment' button
 */
TaskEntry.prototype.start = function() {
    debug("TaskEntry.prototype.start")
    this.disable();
    return this.run(true, true);
}

TaskEntry.prototype.saverec = function() {
    this.disable();
    return this.run(true, false);
}

TaskEntry.prototype.run = function(save, exec) {
    debug("TaskEntry.run")
    // activate the report; start listening to the websocket and update the 'report' field when new data is received
    if (this.report){
        this.report.destroy();
    }
    this.report = new Report(task_interface.trigger.bind(this));
    this.report.activate();
    this.report.set_mode("running");
    this.files.hide();

    var form = {};
    form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
    form['data'] = JSON.stringify(this.get_data());

    // post to different URL depending on whether the data should be saved or not
    var post_url = "";
    if (save && exec) {
        post_url = "/start";
    } else if (save && !exec) {
        post_url = "/saverec";
    } else if (!save && exec) {
        post_url = "/test";
    }

    $.post(post_url, form,
        function(info) {
            task_interface.trigger.bind(this)(info);
            this.report.update(info);
            if (info.status == "running") {
                this.new_row(info);
                this.start_button_pressed = true;
            } else if (info.status == "completed") {
                this.new_row(info);
                this.tr.removeClass("running active error testing")
                this.destroy();
                te = new TaskEntry(this.idx);
                te.tr.addClass("active");
            }
       }.bind(this)
    );
    return false;
}

TaskEntry.prototype.new_row = function(info) {
    //
    // Add a row to the left hand side table. This function is run after you start a new task using the 'Start Experiment' button
    //
    debug('TaskEntry.prototype.new_row: ' + info.idx);

    this.idx = info.idx;
    this.tr.removeClass("running active error testing")

    // make the row hidden (becomes visible when the start or test buttons are pushed)
    this.tr.hide();
    this.tr.click(function() {
        if (te) te.destroy();
        te = new TaskEntry(null);
    })
    //Clean up event bindings
    feats.unbind_change_callback();
    $("#tasks").unbind("change");

    this.tr = $(document.createElement("tr"));

    // add an id number to the row
    this.tr.attr("id", "row"+info.idx);

    // Write the HTML for the table row
    this.tr.html("<td class='colDate'>Now</td>" +
                "<td class='colTime' >--</td>" +
                "<td class='colID'   >"+info.idx+"</td>" +
                "<td class='colSubj' >"+info.subj+"</td>" +
                "<td class='colTask' >"+info.task+"</td>");


    // Insert the new row after the top row of the table
    $("#newentry").after(this.tr);
    this.tr.addClass("active rowactive running");
    this.tr.find('td').addClass("firstRowOfday");
    this.tr.next().find('td').removeClass("firstRowOfday");
    this.notes = new Notes(this.idx);
}

TaskEntry.prototype.get_data = function() {
    var data = {};
    data['subject'] = parseInt($("#subjects").attr("value"));
    data['task'] = parseInt($("#tasks").attr("value"));
    data['feats'] = feats.get_checked_features();
    data['params'] = this.params.to_json();
    data['sequence'] = this.sequence.get_data();
    data['entry_name'] = $("#entry_name").val();
    data['date'] = $("#newentry_today").html();

    return data
}
TaskEntry.prototype.enable = function() {
    debug("TaskEntry.prototype.enable");
    this.params.enable();
    feats.enable_entry();
    if (this.sequence)
        this.sequence.enable();
    if (!this.idx)
        $("#subjects, #tasks").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
    debug("TaskEntry.prototype.disable");
    this.params.disable();
    feats.disable_entry();
    if (this.sequence)
        this.sequence.disable();
    if (!this.idx)
        $("#subjects, #tasks").attr("disabled", "disabled");
}

TaskEntry.prototype.link_new_files = function() {
    data = {
        "data_system_id": $("#data_system_id").val(),
    }

    var file_path = $("#file_path").val();
    var new_file_path = $("#new_file_path").val();
    var new_file_data = $("#new_file_raw_data").val();
    var new_file_data_format= $("#new_file_data_format").val();
    var browser_sel_file = document.getElementById("file_path_browser_sel").files[0];

    if ($.trim(new_file_data) != "" && $.trim(new_file_path) != "") {
        data['file_path'] = new_file_path;
        data['raw_data'] = new_file_data;
        data['raw_data_format'] = new_file_data_format;
    } else if (file_path != "") {
        data['file_path'] = file_path;
        data['raw_data'] = ''
        data['raw_data_format'] = null;
    } else if (browser_sel_file != undefined) {
        data['file_path'] = browser_sel_file.name;
        data['raw_data'] = ''
        data['raw_data_format'] = null;
    } else {
        data = {};
    }

    $.post("/exp_log/link_data_files/" + this.idx + "/submit", data,
        function(resp) {
            $("#file_modal_server_resp").append(resp + "<br>");
            debug("posted the file!");
        }
    )
}

//
// Notes class
//
function Notes(idx) {
    this.idx = idx;
    $("#notes").val("");
    debug("Cleared notes")
    this.activate();
}
Notes.prototype.update = function(notes) {
    //debug("Updating notes to \""+notes+"\"");
    $("#notes textarea").attr("value", notes);
}
Notes.prototype.activate = function() {
    var notes_keydown_handler = function() {
        if (this.last_TO != null)
            clearTimeout(this.last_TO);
        this.last_TO = setTimeout(this.save.bind(this), 500);
    }.bind(this);
    $("#notes textarea").keydown(notes_keydown_handler);
}
Notes.prototype.destroy = function() {
    // unbind the handler to save notes to the database (see 'activate')
    $("#notes textarea").unbind("keydown");

    // clear the timeout handler and save if notes are changing
    if (this.last_TO != null)
        clearTimeout(this.last_TO);
        this.save();
}
Notes.prototype.save = function() {
    this.last_TO = null;
    var notes_data = {
        "notes"                 : $("#notes textarea").attr("value"),
        'csrfmiddlewaretoken'   : $("#experiment input[name=csrfmiddlewaretoken]").attr("value")
    };
    $.post("ajax/save_notes/"+this.idx+"/", notes_data);
    debug("Saved notes");
}

//
// Controls class
//

function create_control_callback(control_str, args) {
    return function() {trigger_control(control_str, args)}
}

function trigger_control(control, params) {
    debug("Triggering control: " + control)
    $.post("trigger_control", {"control": control, "params": JSON.stringify(params.to_json())}, function(resp) {
        debug("Control response", resp)
    })
}

function Controls() {
    this.control_list = [];
    this.params_list = [];
}
Controls.prototype.update = function(controls) {
    debug("Updating controls");
    $("#controls_table").html('');
    this.control_list = [];
    for (var i = 0; i < controls.length; i += 1) {

        var new_params = new Parameters();
        new_params.update(controls[i].params)
        
        var new_button = $('<button/>',
            {
                text: controls[i].name,
                id: "controls_btn_" + i.toString(),
                click: create_control_callback(controls[i].name, new_params),
                type: "button"
            }
        );

        $("#controls_table").append(new_button);
        $("#controls_table").append(new_params.obj)
        this.control_list.push(new_button);
        this.params_list.push(new_params)

    }
    if (this.control_list.length > 0) this.show();
    else this.hide();
}
Controls.prototype.hide = function() {
    $("#controls").hide();
}
Controls.prototype.show = function() {
    if (this.control_list.length > 0) $("#controls").show();
    this.deactivate();
}
Controls.prototype.activate = function() {
    for (var i = 0; i < this.control_list.length; i += 1) {
        $(this.control_list[i]).prop('disabled', false)
    }
    for (var i = 0; i < this.params_list.length; i += 1) {
        this.params_list[i].enable();
    }
}
Controls.prototype.deactivate = function() {
    for (var i = 0; i < this.control_list.length; i += 1) {
        $(this.control_list[i]).prop('disabled', true);
    }
    for (var i = 0; i < this.params_list.length; i += 1) {
        this.params_list[i].disable();
    }
}