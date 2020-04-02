
var log_mode = 5

function log(msg, level) {
    if (level <= log_mode) {
        console.log(msg)
    }
}

function debug(msg) {
    log(msg, 5);
}




var report_activation = null;

//
// TaskInterface class
//
function interface_fn_completed() {
    console.log("state = completed");
    $(window).unbind("unload");
    this.tr.addClass("rowactive active");
    $(".active").removeClass("running error testing");
    this.disable();
    $("#start_buttons").hide()
    $("#stop_buttons").hide();
    $("#finished_task_buttons").show();
    $("#bmi").hide();
    

    $("#report").show()
    $("#notes").show()      

    // Hack fix. When you select a block from the task interface, force the 'date' column to still be white
    if (this.__date) {
        this.__date.each(function(index, elem) {
            $(this).css('background-color', '#FFF');
        });
    }

    if (this.start_button_pressed) {
        setTimeout(
            function () {
                te.report.deactivate();
                clearTimeout(report_activation);
                te = new TaskEntry(te.idx);
            },
            3000
        );
    }
}

function interface_fn_stopped() {
    console.log("state = stopped")
    $(window).unbind("unload");
    $(".active").removeClass("running error testing");
    this.tr.addClass("rowactive active");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    clearTimeout(report_activation);
    $("#bmi").hide();

    $("#report").show()
    $("#notes").hide()
}

function interface_fn_running(info) {
    console.log("state = running")
    $(window).unbind("unload"); // remove any bindings to 'stop' methods when the page is reloaded (these bindings are added in the 'testing' mode)
    $(".active").removeClass("error testing").addClass("running");
    this.disable();
    $("#stop_buttons").show()
    $("#start_buttons").hide();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.activate();

    $("#report").show()
    $("#notes").show()              
}

function interface_fn_testing(info) {
    console.log("state = testing");
    // if you navigate away from the page during 'test' mode, the 'TaskEntry.stop' function is set to run
    $(window).unload(te.stop);

    $(".active").removeClass("error running").addClass("testing");
    te.disable(); // disable editing of the exp_content interface

    $("#stop_buttons").show();
    $("#start_buttons").hide()
    $("#finished_task_buttons").hide()
    $("#bmi").hide();
    this.report.activate();

    $("#report").show();
    $("#notes").hide();
}

function interface_fn_error(info) {
    console.log("state = error");
    $(window).unbind("unload");
    $(".active").removeClass("running testing").addClass("error");
    this.disable();
    $("#start_buttons").hide();
    $("#finished_task_buttons").show();
    $("#bmi").hide();
    this.report.deactivate();
    clearInterval(report_activation);

    $("#report").show()
}
function interface_fn_errtest(info) {
    console.log("state = errtest");

    $(window).unbind("unload");
    $(".active").removeClass("running testing").addClass("error");
    this.enable();
    $("#stop_buttons").hide();
    $("#start_buttons").show();
    $("#finished_task_buttons").hide();
    $("#bmi").hide();
    this.report.deactivate();
    clearInterval(report_activation);

    $("#report").show()
}

function TaskInterfaceConstructor() {
    debug("TaskInterfaceConstructor")
    var state = "";
    var lastentry = null;

    this.trigger = function(info) {
        debug("TaskInterfaceConstructor.trigger");
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
        "completed": interface_fn_completed,
        "stopped": interface_fn_stopped,
        "running": interface_fn_running,
        "testing": interface_fn_testing,
        "error": interface_fn_error,
        "errtest": interface_fn_errtest,
    };
}


var TaskInterface = new TaskInterfaceConstructor();

function create_annotation_callback(annotation_str) {
    return function() {record_annotation(annotation_str)}
}

function record_annotation(annotation) {
    debug("calling record_annotation: " + annotation)
    $.post("record_annotation", {"annotation": annotation}, function(resp) {
        console.log("Annotation response", resp)
    })
}

function Annotations() {
    this.annotation_buttons = [];

    this.update = function(taskinfo) {
        this.destroy_annotation_buttons();
        if (taskinfo.annotations) {
            this.annotations = taskinfo.annotations;
            for (var i = 0; i < taskinfo.annotations.length; i += 1) {
                var new_button = $('<button/>',
                    {
                        text: taskinfo.annotations[i],
                        id: "annotation_btn_" + i.toString(),
                        click: create_annotation_callback(taskinfo.annotations[i]),
                        type: "button"
                    }
                );

                var new_break = $("<br>");

                $("#annot_div").append(new_button);
                $("#annot_div").append(new_break);
                this.annotation_buttons.push(new_button);
                this.annotation_buttons.push(new_break);
            }
        }
    }

    this.update_from_server = function(taskid, sel_feats) {
        $.getJSON("ajax/task_info/"+taskid+"/", sel_feats, 
            function(taskinfo) {
                this.update(taskinfo);
            }.bind(this)
        );
    }

    this.destroy_annotation_buttons = function() {
        for (var i = 0; i < this.annotation_buttons.length; i += 1) {
            this.annotation_buttons[i].remove()
        }
    }

    this.destroy = function() {
        this.destroy_annotation_buttons();
    }

    this.hide = function() {
        $("#annotations").hide();
    }

    this.show = function() {
        $("#annotations").show();
    }
}


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
        if (sys == "sequence") { 
            // Do nothing. No point in showing the sequence..
        } else {  
            // info.datafiles[sys] is an array of files for that system
            for (var i = 0; i < datafiles[sys].length; i++) {
                // Create a list element to hold the file name
                var file = document.createElement("li");
                file.textContent = datafiles[sys][i];
                this.filelist.appendChild(file);
                numfiles++;
            }
        }
    }

    if (numfiles > 0) {
        // Append the files onto the #files field
        $("#file_list").append(this.filelist);

        for (var sys in datafiles)
            if ((sys == "plexon") || (sys == "blackrock") || (sys == "tdt")) {
                this.neural_data_found = true;
                break;
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

    // hide short descriptions
    $('.colShortDesc').hide()

    // hide the old content
    $("#content").hide(); 

    this.sequence = new Sequence();
    this.params = new Parameters();
    this.report = new Report(TaskInterface.trigger.bind(this));
    this.annotations = new Annotations();
    this.files = new Files();
    
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
        if (this.status == 'running'){
            this.report.activate();
        } else {
            this.tr.addClass("rowactive active");
        }

        if (this.status == "completed") {
            this.annotations.hide();
            this.report.set_mode("completed");
            this.files.show();
        }

        // Show the wait wheel before sending the request for exp_info. It will be hidden once data is successfully returned and processed (see below)
        $('#wait_wheel').show();
        $('#tr_seqlist').hide();

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

                // console.log('setting ')
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
    } else { 
        // a "new" task entry is being created
        // this code block executes when you click the header of the left table (date, time, etc.)
        this.idx = null;
        $("#entry_name").val("");

        // show the bar at the top left with drop-downs for subject and task
        this.tr = $("#newentry");
        this.tr.show();  // declared in list.html
        this.status = "stopped";

        // 
        $('#tr_seqlist').show();

        // Set 'change' bindings to re-run the _task_query function if the selected task or the features change
        $("#tasks").change(this._task_query.bind(this));
        feats.bind_change_callback(this._task_query.bind(this))

        if (info) { // if info is present and the id is null, then this block is being copied from a previous block
            console.log('creating a new JS TaskEntry by copy')
            this.update(info);

            // update the annotation buttons
            var taskid = $("#tasks").attr("value");
            var sel_feats = feats.get_checked_features();
            this.annotations.update_from_server(taskid, sel_feats);
            this.enable();
            $("#content").show("slide", "fast");

            this.files.hide();
        } else { // no id and no info suggests that the table header was clicked to create a new block
            console.log('creating a brand-new JS TaskEntry')
            feats.clear();
            this.annotations.hide();
            this.report.hide();
            this.files.hide();
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
    
    this.being_copied = false;
}
/* Populate the 'exp_content' template with data from the 'info' object
 */ 
TaskEntry.prototype.update = function(info) {
    debug("TaskEntry.prototype.update");

    // populate the list of generators
    if (Object.keys(info.generators).length > 0) {
        console.log('limiting generators')
        this.sequence.update_available_generators(info.generators);
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

    console.log("TaskEntry.prototype.update done!");
}
TaskEntry.prototype.reload = function() {
    this.files.clear();

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

            // console.log('setting ')
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

TaskEntry.prototype.save_name = function() {
    $.post("save_entry_name", {"id": this.idx, "entry_name": $("#entry_name").val()});
}

TaskEntry.prototype.toggle_backup = function() {
    debug("TaskEntry.prototype.toggle_backup")
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


/* callback for 'Copy Parameters' button. Note this is not a prototype function
 */
TaskEntry.copy = function() {
    debug("TaskEntry.copy")
    // start with the info saved in the current TaskEntry object
    var info = te.expinfo;

    info.report = {};          // clear the report data
    info.datafiles = {};       // clear the datafile data
    info.notes = "";           // clear the notes
    
    te.being_copied = true;
    te = new TaskEntry(null, info);
    $('#report').hide();        // creating a TaskEntry with "null" goes into the "stopped" state
}

/*
 * Destructor for TaskEntry objects
 */
TaskEntry.prototype.destroy = function() {
    debug("TaskEntry.prototype.destroy")
    $("#content").hide();

    // Destruct the Report object for this TaskEntry
    this.report.destroy();
    
    // Destruct the Sequence object for this TaskEntry 
    if (this.being_copied) {
        // don't destroy when copying because two objects try to manipulate the 
        // Sequence at the same time
        this.sequence.destroy_parameters();
    } else {
        this.sequence.destroy();    
    }

    this.annotations.destroy();

    // Free the parameters
    if (this.params) {
        $(this.params.obj).remove();
        delete this.params;
    }

    // Clear out list of files
    $("#file_list").html("")

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
        feats.unbind_change_callback();
        $("#tasks").unbind("change");
    }
}

TaskEntry.prototype._task_query = function(callback) {
    debug('TaskEntry.prototype._task_query')
    var taskid = $("#tasks").attr("value");
    var sel_feats = feats.get_checked_features();

    $.getJSON("ajax/task_info/"+taskid+"/", sel_feats, 
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

            this.annotations.update(taskinfo);

            if (taskinfo.generators) {
                this.sequence.update_available_generators(taskinfo.generators);
            }
        }.bind(this)
    );
}

TaskEntry.prototype.stop = function() {
    /* Callback for the "Stop Experiment" button
    */
    debug("TaskEntry.prototype.stop")
    var csrf = {};
    csrf['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
    $.post("stop/", csrf, TaskInterface.trigger.bind(this));
    console.log("TaskEntry.prototype.stop")
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
    this.report = new Report(TaskInterface.trigger.bind(this)); // TaskInterface.trigger is a function. 
    this.report.activate();
    this.report.set_mode("running");

    this.annotations.show();
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
            TaskInterface.trigger.bind(this)(info);
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
        te = new TaskEntry(null);
    })
    //Clean up event bindings
    feats.unbind_change_callback();
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
    data['feats'] = feats.get_checked_features();
    data['params'] = this.params.to_json();
    data['sequence'] = this.sequence.get_data();
    data['entry_name'] = $("#entry_name").val();
    data['date'] = $("#newentry_today").html();

    return data
}
TaskEntry.prototype.enable = function() {
    debug("TaskEntry.prototype.enable");
    $("#parameters input").removeAttr("disabled");
    feats.enable_entry();
    if (this.sequence)
        this.sequence.enable();
    if (!this.idx)
        $("#subjects, #tasks").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
    debug("TaskEntry.prototype.disable");
    $("#parameters input").attr("disabled", "disabled");
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
    var new_file_data_format= $("new_file_data_format").val();
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
            console.log("posted the file!");
        }
    )
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
    var notes_keydown_handler = function() {
        if (this.last_TO != null)
            clearTimeout(this.last_TO);
        this.last_TO = setTimeout(this.save.bind(this), 2000);
    }.bind(this);
    $("#notes textarea").keydown(notes_keydown_handler);
}
Notes.prototype.destroy = function() {
    // unbind the handler to save notes to the database (see 'activate')
    $("#notes textarea").unbind("keydown");    

    // clear the text
    $("#notes").val("");

    // clear the timeout handler
    if (this.last_TO != null)
        clearTimeout(this.last_TO);

    // save right at the end
    this.save();
}
Notes.prototype.save = function() {
    this.last_TO = null;
    var notes_data = {
        "notes"                 : $("#notes textarea").attr("value"), 
        'csrfmiddlewaretoken'   : $("#experiment input[name=csrfmiddlewaretoken]").attr("value")
    };
    $.post("ajax/save_notes/"+this.idx+"/", notes_data);
}