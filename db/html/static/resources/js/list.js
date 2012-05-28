function TaskEntry(idx, info){
	this.sequence = new Sequence();
	this.params = new Parameters();
	this.report = new Report(idx == null, this._update_status.bind(this));
		
	$("#parameters").append(this.params.obj);

	if (idx) {
		this.idx = parseInt(idx.match(/row(\d+)/)[1]);
		this.tr = $("#"+idx);
		$("#copybtn").show();
		$("#startbtn, #testbtn").hide();
		$.getJSON("ajax/exp_info/"+this.idx+"/", {}, function (expinfo) {
			this.update(expinfo);
			this.disable();
			$("#content").show("slide", "fast");
		}.bind(this));
	} else {
		this.idx = null;
		this.tr = $("#newentry").show();
		$("#tasks").change(this._task_query.bind(this));
		$("#features input").change(this._task_query.bind(this));
		$("#copybtn").hide();
		$("#startbtn, #testbtn").show();
		if (info) {
			this.update(info);
			this.enable();
			$("#content").show("slide", "fast");
		} else {
			this._task_query(function() {
				this.enable();
				$("#content").show("slide", "fast");
			}.bind(this));
		}
	}

	this.tr.unbind("click");
	this.tr.addClass("rowactive active");
}
TaskEntry.prototype._update_status = function(info) {
	console.log(info);
	if (info.status == "running" || info.status == "testing") {
		this.disable();
		$("#testbtn").hide();
		$("#startbtn").attr("value", "Stop");
		$("#experiment").unbind("submit").submit(this.stop.bind(this));
	}

	if (info.status == "running") {
		$(".active").addClass("running");
	} else if (info.status == "testing" && info.state != null) {
		$(".active").removeClass("running error").addClass("testing");
	} else if (info.status == "error") {
		$(".active").removeClass("running testing").addClass("error");
	}

	if (info.status == "testing" && info.state == null || info.status == "stopped") {
		$(".active").removeClass("running error testing");
		this.enable();
		$("#testbtn").show();
		$("#startbtn").attr("value", "Start Experiment");
		$("#experiment").unbind("submit").submit(this.run.bind(this, btn));
	}
}

TaskEntry.prototype.update = function(info) {
	this.expinfo = info;
	$("#tasks option").each(function() {
		if (this.value == info.task)
			this.selected = true;
	})
	$("#subjects option").each(function() {
		if (this.value == info.subject)
			this.selected = true;
	});
	$("#features input[type=checkbox]").each(function() {
		this.checked = false;
		for (var idx in info.feats) {
			if (this.name == info.feats[idx])
				this.checked = true;
		}
	});
	this.sequence.update(info.sequence);
	this.params.update(info.params);
	this.report.update(info.report);

	if (info.sequence) {
		$("#sequence").show()
	} else {
		$("#sequence").hide()
	}
}
TaskEntry.copy = function() {
	te.destroy();
	te.expinfo.report = {};
	te = new TaskEntry(null, te.expinfo);
}
TaskEntry.prototype.destroy = function() {
	$("#content").hide();
	this.report.destroy();
	this.sequence.destroy();
	$(this.params.obj).remove()
	delete this.params
	this.tr.removeClass("rowactive active");

	if (this.idx != null) {
		var idx = "row"+this.idx;
		this.tr.click(function() {
			if (te) te.destroy();
			te = new TaskEntry(idx);
		})
	} else {
		//Remove the newentry row
		this.tr.hide()
		//Rebind the click action
		this.tr.click(function() {
			if (te) te.destroy();
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

TaskEntry.prototype.run = function(save) {
	this.disable();
	var form = {};
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(this.get_data());
	$.post(save?"start":"test", form, function(info) {
		this.report.update(info);
		this._update_status(info);
	}.bind(this));
	return false;
}
TaskEntry.prototype.stop = function(data) {
	$(".active").removeClass("running");
	this.tr.removeClass("running");
	$("#content .startbtn").hide()
	$("#addbtn").show()
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
