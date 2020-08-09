// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(`    <fieldset id="report">
          <legend>Report</legend>
          <div id="report_div">
            <input type="button" value="Update report" id="report_update" onclick="$.post('report', {}, function(info) {te.report.update(info['data']); console.log(info);})"><br>
            <table class="option" id="report_info">
            </table>

            <div class="report_table" id="report_msgs">
              <pre id="stdout"></pre>
            </div>


            <div class="clear"></div>
          </div>
        </fieldset>    `);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}


function Report(callback) {
    // store a ref to the callback function passed in
    this.notify = callback;

    // this.info is a summary stat table
    this.info = $("#report_info");

    // this.msgs = text printed by the task
    this.msgs = $("#report_msgs");

    // Used for error messages?
    this.stdout = $("#stdout");

    this.boxes = {};
    this.ws = null;
}

Report.prototype.activate = function() {
    var on_windows = window.navigator.userAgent.indexOf("Windows") > 0;
    if (!this.ws && !on_windows) {  // some websocket issues on windows..
        // Create a new JS WebSocket object
        this.ws = new WebSocket("ws://"+hostname.split(":")[0]+":8001/connect");

        this.ws.onmessage = function(evt) {
            console.log(evt.data);
            var report = JSON.parse(evt.data);
            this.update(report);
        }.bind(this);
    }
}

Report.prototype.update = function(info) {
    // run the 'notify' callback every time this function is provided with info
    if (typeof(this.notify) == "function" && info)
        this.notify(info);

    if (info.status && info.status == "error") { // received an error message through the websocket
        // append the error message (pre-formatted by python traceback) onto the printed out messages
        this.msgs.append("<pre>"+info.msg+"</pre>");
    } else if (info.status && info.status == "stdout") {
        this.stdout.append(info.msg);
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

                    this.info.append(row);

                    // save ref to the 'data' box, to be updated when new 'info' comes in
                    this.boxes[stat] = data;
                }
            }
        }

        // Update the stat data
        for (var stat in this.boxes) {
            if (info[stat])
                this.boxes[stat].innerHTML = info[stat];
        }
    }
}
Report.prototype.manual_update = function() {
    $.post('report', {}, function(info) {te.report.update(info['data']);});
}

Report.prototype.destroy = function () {
    this.deactivate();
    this.msgs.html('<pre id="stdout"></pre>');
    this.info.html("");
}

Report.prototype.deactivate = function() {
    /*
        Close the report websocket
    */
    if (this.ws)
        this.ws.close();
    delete this.ws;
}
Report.prototype.hide = function() {
    $("#report").hide();
}

Report.prototype.show = function() {
    $("#report").show();
}

Report.prototype.set_mode = function(mode) {
    if (mode == "completed") {
        $("#report_update").hide();
    } else if (mode == "running") {
        $("#report_update").show();
    }
}

if (typeof(module) !== 'undefined' && module.exports) {
  exports.Report = Report;
  exports.$ = $;
}
