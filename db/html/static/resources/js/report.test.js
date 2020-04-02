QUnit.test("Report tests", function(assert) {
	var callback_info = null;
	var report = new Report(function(info) {callback_info = info});
	
	// test update of error messages
	var error_report = {"status":"error", "msg":"Test error message"}
	report.update(error_report);
	assert.equal(report.msgs.html().endsWith("<pre>Test error message</pre>"), true);
	assert.equal(error_report, callback_info)

	// test update of regular print statements
	var print_report = {"status":"stdout", "msg":"Test print statement"}
	report.update(print_report);
	assert.equal(report.stdout.html().endsWith("Test print statement"), true);
	assert.equal(print_report, callback_info)

	// test update of table stats
	var stat_report = {"stat1":"things", "stat2":"more things"};
	report.update(stat_report);
	var report_table_matches = $("#report_info")[0].innerHTML.endsWith(
		"<tbody><tr><td>stat1</td><td>things</td></tr><tr><td>stat2</td><td>more things</td></tr></tbody>")
	assert.equal(report_table_matches, true);
	assert.equal(stat_report, callback_info)

	// TODO add test for callback

	// TODO test hiding of update button
	report.set_mode("completed");

	report.set_mode("running");	
});