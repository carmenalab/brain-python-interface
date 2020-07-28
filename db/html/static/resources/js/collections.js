function Collections() {
    this.collections_obj = $("#collections");
}

Collections.prototype.hide = function() {
    this.collections_obj.hide();
}

Collections.prototype.show = function() {
    this.collections_obj.show();
}

Collections.prototype.select_collections = function(collections) {
    // clear
    console.log(collections);
    $("#collections input[type=checkbox]").each(
        function() {
            this.checked = false;
            for (var idx in collections) {
                if (this.name == collections[idx]) {
                    this.checked = true;
                }
            }
       }
    );
}

Collections.prototype.update_collection_membership = function(val) {

}

// /* 
//  * Uncheck all the features
//  */
// Features.prototype.clear = function() {
//     $("#features input[type=checkbox]").each(
//         function() {
//             this.checked = false;
//        }
//     );
// }
// /* 
//  * Check boxes for a set of features
//  */
// Features.prototype.select_features = function(info_feats) {
//     // set checkmarks for all the features specified in the 'info'
//     $("#features input[type=checkbox]").each(
//         function() {
//             this.checked = false;
//             for (var idx in info_feats) {
//                 if (this.name == info_feats[idx])
//                     this.checked = true;
//             }
//        }
//     );    
// }
// /* 
//  * Get a JSON object with the list of enabled features. Keys = feature names, 
//  * values are 'true'
//  */
// Features.prototype.get_checked_features = function () {
//     var feats = {};
//     $("#features input").each(function() { 
//         if (this.checked) 
//             feats[this.name] = this.checked;    
//     });
//     return feats;    
// }
// /* 
//  * Disable the ability to check/uncheck features
//  */
// Features.prototype.disable_entry = function () {
//     $("#features input").attr("disabled", "disabled");
// }
// /* 
//  * Enable the ability to check features
//  */
// Features.prototype.enable_entry = function() {
//     $("#features input").removeAttr("disabled");
// }
// /* 
//  * If any of the features are chagned or unchanged, run the callback
//  */
// Features.prototype.bind_change_callback = function(callback) {
//     $("#features input").change(callback);
// }
// /* 
//  * Disable any callbacks assigned by 'bind_change_callback'
//  */
// Features.prototype.unbind_change_callback = function() {
//     $("#features input").unbind("change");
// }