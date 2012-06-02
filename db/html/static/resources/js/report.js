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
function Report(callback) {
    this.notify = callback;
    this.obj = document.createElement("div");
    this.info = document.createElement("table");
    this.msgs = document.createElement("div");
    this.info.className = "options";
    this.msgs.className = "rightside";
    this.obj.appendChild(this.info);
    this.obj.appendChild(this.msgs);
    $("#report").append(this.obj);

    this.boxes = {"state":"Current State", "trials":"Trial #", "length":"Time", "reward_len":"Reward Time", "rates":"Rates"};
    for (var i in this.boxes) {
        var row = document.createElement("tr");
        var label = document.createElement("td");
        label.innerHTML = this.boxes[i];
        row.appendChild(label);
        var data = document.createElement("td");
        row.appendChild(data);
        this.info.appendChild(row);
        this.boxes[i] = data;
    }
}
Report.prototype.activate = function() {
    this.ws = new WebSocket("ws://"+hostname.split(":")[0]+":8001/connect");
    this.ws.onmessage = function(evt) {
        var report = JSON.parse(evt.data);
        if (report.trials)
            report.trials++;
        this.update(report);
    }.bind(this);
}
Report.prototype.deactivate = function() {
    if (this.ws)
        this.ws.close();
    delete this.ws;
}
Report.prototype.update = function(info) {
    if (typeof(this.notify) == "function" && info)
        this.notify(info);
    if (info.status && info.status == "error") {
        this.msgs.innerHTML = "<pre>"+info.msg+"</pre>";
    } else {
        for (var i in this.boxes) {
            if (info[i])
                if (box_filters[i])
                    this.boxes[i].innerHTML = box_filters[i](info[i]);
                else
                    this.boxes[i].innerHTML = info[i];
        }
    }
}
Report.prototype.destroy = function () {
    this.deactivate();
    this.obj.parentNode.removeChild(this.obj);
}