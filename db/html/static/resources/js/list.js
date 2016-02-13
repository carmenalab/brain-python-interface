//
// TaskInterface class
//
function TaskInterfaceConstructor() {
	var state = "";
	var lastentry = null;

    this.trigger = function(info) {
		if (this != lastentry) {
			if (lastentry) {
				$(window).unload(); //This stops testing runs, just in case
				lastentry.tr.removeClass("rowactive active");
				lastentry.destroy();
			}
			states[this.status].bind(this)(info);
			lastentry = this;
			state = this.status;
		}
		for (var next in triggers[state]) {
			if (triggers[state][next].bind(this)(info)) {
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
	var states = {
		completed: function() {
			$(window).unbind("unload");
			this.tr.addClass("rowactive active");
			$(".active").removeClass("running error testing");
			this.disable();
			$(".startbtn").hide()
			$("#copybtn").show();
			$("#bmi").hide();
			this.report.deactivate();
		},
		stopped: function() {
			$(window).unbind("unload");
			$(".active").removeClass("running error testing");
			this.tr.addClass("rowactive active");
			this.enable();
			$("#stopbtn").hide()
			$("#startbtn").show()
			$("#testbtn").show()
			$("#copybtn").hide();
			$("#bmi").hide();
		},
		running: function(info) {
			$(window).unbind("unload");
			$(".active").removeClass("error testing").addClass("running");
			this.disable();
			$("#stopbtn").show()
			$("#startbtn").hide()
			$("#testbtn").hide()
			$("#copybtn").hide();
			$("#bmi").hide();
			// this.report.activate();
		},
		testing: function(info) {
			$(window).unload(this.stop.bind(this));
			$(".active").removeClass("error running").addClass("testing");
			this.disable();
			$("#stopbtn").show()
			$("#startbtn").hide()
			$("#testbtn").hide()
			$("#copybtn").hide()
			$("#bmi").hide();
			// this.report.activate();
		},
		error: function(info) {
			$(window).unbind("unload");
			$(".active").removeClass("running testing").addClass("error");
			this.disable();
			$(".startbtn").hide();
			$("#copybtn").show();
			$("#bmi").hide();
			this.report.deactivate();
		},
		errtest: function(info) {
			$(window).unbind("unload");
			$(".active").removeClass("running testing").addClass("error");
			this.enable();
			$("#stopbtn").hide();
			$("#startbtn").show();
			$("#testbtn").show();
			$("#copybtn").hide();
			$("#bmi").hide();
			this.report.deactivate();
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

	if (idx) { // the task entry which was clicked has an id (stored in the database)

		// parse the actual integer database ID out of the HTML object name
		this.idx = parseInt(idx.match(/row(\d+)/)[1]);

		// Create a jQuery object to represent the table row
		this.tr = $("#"+idx);

		this.status = this.tr.hasClass("running") ? "running" : "completed";
		if (this.status == 'running')
			this.report.activate();

		$.getJSON("ajax/exp_info/"+this.idx+"/", // URL to query for data on this task entry
			{}, // POST data to send to the server
			function (expinfo) { // function to run on successful response
				this.notes = new Notes(this.idx);
				this.update(expinfo);
				this.disable();
				$("#content").show("slide", "fast");
			}.bind(this));
	} else { // a "new" task entry is being created
		this.idx = null;

		// show the bar at the top left with drop-downs for subject and task
		this.tr = $("#newentry").show();  // declared in list.html
		this.status = "stopped";
		$("#tasks").change(this._task_query.bind(this));
		$("#features input").change(this._task_query.bind(this));

		if (info) {
			// if the 'info' is provided, 
			this.update(info);
			this.enable();
			$("#content").show("slide", "fast");
		} else {
			console.log("flagged for backup " + info.flagged_for_backup)

			TaskInterface.trigger.bind(this)({state:''});
			this._task_query(function() {
				this.enable();
				$("#content").show("slide", "fast");
			}.bind(this));
		}
		$("#notes textarea").val("").removeAttr("disabled");
	}
	
	// Disable reacting to clicks on the current row so that things don't get reset
	this.tr.unbind("click");
}

TaskEntry.prototype.new_row = function(info) {
    /* Add a row to ..something
     */
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
	this.tr.attr("id", "row"+info.idx);
	this.tr.html("<td class='colDate'>"+info.date+" ("+info.idx+")</td>" + 
		"<td class='colSubj'>"+info.subj+"</td>" + 
		"<td class='colTask'>"+info.task+"</td>");

	// Insert the new row after the top row of the table
	$("#newentry").after(this.tr);
	this.tr.addClass("active rowactive running");
	this.notes = new Notes(this.idx);
}

/* Populate the 'exp_content' template with data from the 'info' object
 */ 
TaskEntry.prototype.update = function(info) {
	// populate the list of generators
	$('#seqgen').empty();
	$.each(info.generators, function(key, value) {   
	     $('#seqgen')
	          .append($('<option>', { value : key })
	          .text(value)); 
	});

	$('#report_backup').html('Flagged for backup: ' + info.flagged_for_backup+"\n<br><br>");

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
	$("#features input[type=checkbox]").each(function() {
		this.checked = false;
		for (var idx in info.feats) {
			if (this.name == info.feats[idx])
				this.checked = true;
		}
	});
	var numfiles = 0;
	this.filelist = document.createElement("ul");
	
	// List out the data files in the 'filelist'
	// see TaskEntry.to_json in models.py to see how the file list is generated
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
		var found = false;
		for (var sys in info.datafiles)
			if ((sys == "plexon") || (sys == "blackrock") || (sys == "tdt")) {
				found = true;
				break;
			}

		if (found)
			// Create the BMI object
			this.bmi = new BMI(this.idx, info.bmi, info.notes);
	}

	if (info.sequence) {
		$("#sequence").show()
	} else {
		// $("#sequence").hide()
	}

    // render plots
	this.plot_filelist = document.createElement("ul");
    if (info.plot_files) {
        $("#plots").empty()
        for (var plot_type in info.plot_files) {
            var img = document.createElement("img");
            img.setAttribute('src', '/static'+info.plot_files[plot_type])
            $("#plots").append(img)
        }
    } else {
        $("#plots").empty()
    }
}
TaskEntry.plot_performance = function() {

}
/* callback for 'Copy Parameters' button
 */
TaskEntry.copy = function() {
	var info = te.expinfo;
	info.report = {};
	info.datafiles = {};
	info.notes = "";
	info.plot_files = {};
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
		$(this.params.obj).remove()
		delete this.params
	}

    // Remove any designations that this TaskEntry is active/running/errored/etc.
	this.tr.removeClass("rowactive active error");
	$("#content").removeClass("error running testing")

    // Hide the 'files' field
	$("#files").hide();
	$(this.filelist).remove();

	if (this.idx != null) {
		var idx = "row"+this.idx;
		this.tr.click(function() {
			te = new TaskEntry(idx);
		})

        // clear the notes field
		this.notes.destroy();

        // clear the BMI
		if (this.bmi !== undefined) {
			this.bmi.destroy();
			delete this.bmi;
		}

	} else {
		//Remove the newentry row
		this.tr.hide()
		//Rebind the click action to create a blank TaskEntry form
		this.tr.click(function() {
			te = new TaskEntry(null);
		})
		//Clean up event bindings
		$("#features input").unbind("change");
		$("#tasks").unbind("change");
	}
}
TaskEntry.prototype._task_query = function(callback) {
	var taskid = $("#tasks").attr("value");
	var feats = {};
	$("#features input").each(function() { 
		if (this.checked) 
			feats[this.name] = this.checked;	
	});

	$.getJSON("ajax/task_info/"+taskid+"/", feats, function(taskinfo) {
		this.params.update(taskinfo.params);
		if (taskinfo.sequence) {
			$("#sequence").show()
			this.sequence.update(taskinfo.sequence);
		} else
			$("#sequence").hide()
		if (typeof(callback) == "function")
			callback();
	}.bind(this));
}

/* Callback for the 'Start experiment' button
 */
TaskEntry.prototype.start = function() {
	this.disable();
	return this.run(true);
}

/* Callback for the 'Test' button
 */
TaskEntry.prototype.test = function() {
	this.disable();
	return this.run(false); 
}

/* Callback for the "Stop Experiment" button
 */
TaskEntry.prototype.stop = function() {
	var csrf = {};
	csrf['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value");
	$.post("stop", csrf, TaskInterface.trigger.bind(this));

	function f() {
		$.post("next_exp/", {
			'csrfmiddlewaretoken':$("#experiment input[name=csrfmiddlewaretoken]").attr("value")
		});
	}
	setTimeout(f, 5000);
}
TaskEntry.prototype.run = function(save) {
    // activate the report; start listening to the websocket and update the 'report' field when new data is received
	this.report.activate();
	var form = {};
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(this.get_data());
	this.report.pause();
	var post_url = save ? "/start" : "/test";
	$.post(post_url, form, function(info) {
		TaskInterface.trigger.bind(this)(info);
		this.report.update(info);
		if (info.status == "running")
			this.new_row(info);
		this.report.unpause();
	}.bind(this));
	return false;
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
