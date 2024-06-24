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
    // $("#bmi").hide();
    
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
            testing: function(info) { return info.status == "testing"; },
            stopped: function(info) { return info.status == "stopped"; },
        },
        "errtest": {
            running: function(info) { return info.status == "running"; },
            testing: function(info) { return info.status == "testing"; },
            stopped: function(info) { return info.status == "stopped"; },
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
            if (sys == "plexon" || sys == "blackrock" || sys == "ecube") {
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
    $('#te_table_header').unbind("click");
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
    this.metadata = new Metadata();
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
        
        feats.clear();
        this.report.hide();
        this.files.hide();
        // task_interface.trigger.bind(this)({state:''});
        
        // query the server for information about the task (which generators can be used, which parameters can be set, etc.)
        this._task_query(
            function() {
                this.enable();
                $("#content").show("slide", "fast");
            }.bind(this), true, true
        );

        // Set 'change' bindings to re-run the _task_query function if the selected task or the features change
        $("#tasks").change(this._task_query.bind(this));
        feats.bind_change_callback(this._task_query.bind(this))

        // make the notes blank and editable
        $("#notes textarea").val("").removeAttr("disabled");

        // Disable reacting to clicks on the current row so that the interface doesn't get reset
        this.tr.unbind("click");
        $('te_table_header').unbind("click");

        task_interface.trigger.bind(this)({status: this.status});
    }
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
    this.metadata.update(info.metadata);
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

    // set the 'tasks' drop-down menu to match the 'info'
    $("#tasks option").each(function() {
        if (this.value == info.task)
            this.selected = true;
    })

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
    var idx = this.idx;
    var name = $("#entry_name").val()
    var tr = this.tr;
    if (this.idx) $.post("save_entry_name", {"id": this.idx, "entry_name": name}, function() {
        if (name) tr.find("td.colID").html(name+" ("+idx+")");
        else tr.find("td.colID").html(idx);
    });
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


/* callback for 'Copy Parameters' button.
 */
TaskEntry.prototype.copy = function() {
    debug("TaskEntry.copy")

    // reset the task entry row
    var idx = this.idx
    this.tr.click(
        function() {
            if (te) te.destroy();
            te = new TaskEntry(idx);
        }
    );
    this.tr.removeClass("rowactive active error");
    this.tr = $("#newentry");
    this.tr.show();
    $('te_table_header').unbind("click");
    this.tr.unbind("click");

    // bind callbacks to the tasks and features fieldsets
    $("#tasks").change(this._task_query.bind(this));
    feats.bind_change_callback(this._task_query.bind(this))

    // reset the task info
    this.idx = null;           // reset the id
    this.status = "stopped";   // set the status
    this.report.destroy();     // clear the report data
    this.files.clear();        // clear the datafile data
    this.files.hide();
    this.notes.destroy();      // clear the notes
    this.report.hide();        // turn off the report pane

    // update the task info, but leave the parameters alone
    this._task_query(function(){}, false, true);

    // go into the "stopped" state
    task_interface.trigger.bind(this)({status: this.status}); 
}

/*
 * Destructor for TaskEntry objects
 */
TaskEntry.prototype.destroy = function() {
    debug("TaskEntry.prototype.destroy")
    $("#content").hide();

    // Destruct objects
    this.report.destroy();
    this.sequence.destroy();
    if (this.params)  $(this.params.obj).remove();
    if (this.notes) this.notes.destroy();
    if (this.bmi) this.bmi.destroy();
    $(this.filelist).remove();

    // Remove any designations that this TaskEntry is active/running/errored/etc.
    this.tr.removeClass("rowactive active error");

    // Re-bind a callback to when the row is clicked
    var idx = this.idx
    this.tr.unbind("click");
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

TaskEntry.prototype._task_query = function(callback, update_params=true, update_metadata=false) {
    debug('TaskEntry.prototype._task_query')
    var taskid = $("#tasks").attr("value");
    var sel_feats = feats.get_checked_features();

    $.getJSON("ajax/task_info/"+taskid+"/", sel_feats,
        function(taskinfo) {
            debug("Information about task received from the server");
            debug(taskinfo);

            if (update_params) this.params.update(taskinfo.params);
            if (typeof(callback) == "function")
                callback();

            if (taskinfo.generators) {
                this.sequence.update_available_generators(taskinfo.generators);
            }
            if (taskinfo.sequence) {
                $("#sequence").show()
                this.sequence.update(taskinfo.sequence);
            } else
                $("#sequence").hide()

            if (update_metadata) this.metadata.update(taskinfo.metadata);

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

// Callback for the 'Test' button
TaskEntry.prototype.test = function() {
    debug("TaskEntry.prototype.test")
    return this.run(false, true);
}

// Callback for the 'Start experiment' button
TaskEntry.prototype.start = function() {
    debug("TaskEntry.prototype.start")
    return this.run(true, true);
}

// Callback for 'Save Record' button
TaskEntry.prototype.saverec = function() {
    return this.run(true, false);
}

TaskEntry.prototype.run = function(save, exec) {
    debug("TaskEntry.run")
    // make sure we're stopped
    task_interface.trigger.bind(this)({status: "stopped"});

    // check that inputs have been filled out
    let valid = true;
    $('[required]').each(function() {
        if ($(this).is(':invalid') || !$(this).val()) valid = false;
    })
    if (!valid) {
        $("#experiment").trigger("submit"); // this will pop up a message to fill out the missing fields
        return;
    }

    // activate the report; start listening to the websocket and update the 'report' field when new data is received
    if (this.report){
        this.report.destroy();
    }
    this.report = new Report(task_interface.trigger.bind(this));
    this.report.activate();
    this.report.set_mode("running");
    this.files.hide();
    this.disable();

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

    if (typeof(info.idx) == "number") {
        this.idx = info.idx;
    } else {
        this.idx = parseInt(info.idx.match(/(\d+)/)[1]);
    }
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
    this.tr.attr("id", "row"+this.idx);

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
    data['task'] = parseInt($("#tasks").attr("value"));
    data['feats'] = feats.get_checked_features();
    data['params'] = this.params.to_json();
    data['metadata'] = this.metadata.get_data();
    data['sequence'] = this.sequence.get_data();
    data['entry_name'] = $("#entry_name").val();
    data['date'] = $("#newentry_today").html();

    return data
}
TaskEntry.prototype.enable = function() {
    debug("TaskEntry.prototype.enable");
    this.params.enable();
    this.metadata.enable();
    feats.enable_entry();
    if (this.sequence)
        this.sequence.enable();
    if (!this.idx)
        $("#subjects, #tasks").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
    debug("TaskEntry.prototype.disable");
    this.params.disable();
    this.metadata.disable()
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
// Metadata class
//
function Metadata() {
    $("#metadata_table").html("")
    var params = new Parameters();
    this.params = params;
    $("#metadata_table").append(this.params.obj);
    var add_new_row = $('<input id="paramadd" type="button" value="+"/>');
    add_new_row.on("click", function() {params.add_row();});
    this.add_new_row = add_new_row;
    $("#metadata_table").append(add_new_row);
}
Metadata.prototype.update = function(info) {
    this.params.update(info)
}
Metadata.prototype.enable = function() {
    this.params.enable();
    this.add_new_row.show();
}
Metadata.prototype.disable = function() {
    this.params.disable();
    this.add_new_row.hide();
}
Metadata.prototype.get_data = function () {
    var data = this.params.to_json();
    return data;
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

    // reset the textarea
    $("#notes textarea").val("").removeAttr("disabled"); 
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

function create_control_callback(i, control_str, args, static=false) {
    return function() {trigger_control(i, control_str, args, static)}
}

function trigger_control(i, control, params, static) {
    debug("Triggering control: " + control)
    if (static) {
        var data = {
            "control": control, 
            "params": JSON.stringify(params.to_json()), 
            "base_class": $('#tasks').val(),
            "feats": JSON.stringify(feats.get_checked_features())
        }
        $.post("trigger_control", data, function(resp) {
            debug("Control response", resp);
            if (resp["status"] == "success") {
                $('#controls_btn_' + i.toString()).css({"background-color": "green"});
                $('#controls_btn_' + i.toString()).animate({"background-color": "black"}, 500 );
            }
        })
    } else {
        $.post("trigger_control", {"control": control, "params": JSON.stringify(params.to_json())}, function(resp) {
            debug("Control response", resp);
            params.clear_all();
            if (resp["status"] == "pending") {
                $('#controls_btn_' + i.toString()).css({"background-color": "yellow"});
                $('#controls_btn_' + i.toString()).animate({"background-color": "black"}, 500 );
            }
        })
    }
}

function Controls() {
    this.control_list = [];
    this.static_control_list = [];
    this.params_list = [];
    this.static_params_list = [];
}
Controls.prototype.update = function(controls) {
    debug("Updating controls");
    $("#controls_table").html('');
    this.control_list = [];
    this.static_control_list = [];
    this.params_list = [];
    this.static_params_list = [];
    for (var i = 0; i < controls.length; i += 1) {

        var new_params = new Parameters();
        new_params.update(controls[i].params)
        
        var new_button = $('<button/>',
            {
                text: controls[i].name,
                id: "controls_btn_" + i.toString(),
                click: create_control_callback(i, controls[i].name, new_params, controls[i].static),
                type: "button"
            }
        );

        $("#controls_table").append(new_button);
        $("#controls_table").append(new_params.obj)

        if (controls[i].static) { // static controls are always active
            this.static_control_list.push(new_button);
            this.static_params_list.push(new_params)
        } else {
            this.control_list.push(new_button);
            this.params_list.push(new_params)
        }

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