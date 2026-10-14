// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    Features = require('../features.js').Features;
    var $ = require('../features.js').$;
}

QUnit.test("Features manipulation", function(assert) {
  var feats = new Features();
  // clear to start
  feats.clear();
  assert.equal($.isEmptyObject(feats.get_checked_features()), true);

  // check 1 feature
  feats.select_features(["feature1"])
  sel_feats = feats.get_checked_features();
  assert.equal(Object.keys(sel_feats).length, 1);
  assert.equal(sel_feats["feature1"], true);

  // test callback on change
  var feats_callback_var = 0;
  feats.bind_change_callback(function() {feats_callback_var += 1;});
  $("#feat_feature1").trigger("click");
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);

  // if you disable the features, the callback should not trigger
  feats.disable_entry();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);

  feats.unbind_change_callback();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2); // no change because disabled and unbound

  // clear again
  feats.clear();
  feats.enable_entry();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);  // no increment because callback should be unbound

  sel_feats = feats.get_checked_features();
  assert.equal(Object.keys(sel_feats).length, 1);
  assert.equal(sel_feats["feature1"], true);
});

QUnit.test("Unavailable features", function(assert) {
  var feats = new Features();
  feats.clear();

  // an old entry that used an unavailable feature still shows it as checked
  feats.select_features(["feature1", "feature3"]);
  assert.equal($("#feat_feature3").prop("checked"), true);

  // but it is never sent to the server
  sel_feats = feats.get_checked_features();
  assert.equal(Object.keys(sel_feats).length, 1);
  assert.equal(sel_feats["feature1"], true);

  // enabling entry (e.g. copying the entry) leaves it disabled and unchecked
  feats.disable_entry();
  feats.enable_entry();
  assert.equal($("#feat_feature1").prop("disabled"), false);
  assert.equal($("#feat_feature3").prop("disabled"), true);
  assert.equal($("#feat_feature3").prop("checked"), false);
  assert.equal($("#feat_feature1").prop("checked"), true);
});
