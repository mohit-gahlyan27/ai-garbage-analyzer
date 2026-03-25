from flask import Flask, render_template, request
import os
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load AI model
model = YOLO("yolov8n.pt")


# Garbage tips dictionary (20+ items)

tips = {

"chair": (
"Furniture",
"Wooden or plastic chairs can often be repaired and reused instead of being discarded. "
"If the chair is still in good condition, consider donating it to schools or charities. "
"Broken chairs can be dismantled and their materials recycled. "
"This reduces landfill waste and promotes sustainable reuse."
),

"bottle": (
"Plastic Waste",
"Plastic bottles should be cleaned and separated before recycling. "
"They can also be reused for storing water, gardening tools, or DIY crafts. "
"Sending them to plastic recycling centers helps reduce environmental pollution. "
"Recycling plastic saves energy and reduces the need for new plastic production."
),

"cup": (
"Plastic Waste",
"Plastic cups can be washed and reused for storage or gardening. "
"They can also be used for planting small plants or seedlings. "
"If they are damaged, send them to a plastic recycling facility. "
"Proper recycling helps reduce plastic waste in oceans and landfills."
),

"banana": (
"Food Waste",
"Banana peels are excellent for composting and can produce nutrient-rich fertilizer. "
"They can also be soaked in water for 24 hours to create liquid fertilizer for plants. "
"Composting food waste improves soil fertility and reduces landfill waste. "
"This process supports sustainable agriculture and eco-friendly waste management."
),

"apple": (
"Food Waste",
"Apple scraps can be composted to create organic fertilizer. "
"They break down quickly and enrich soil nutrients. "
"Food waste composting reduces methane emissions from landfills. "
"It also helps maintain a healthy garden ecosystem."
),

"carrot": (
"Food Waste",
"Carrot waste can be added to compost bins to create natural manure. "
"Vegetable scraps decompose quickly and improve soil health. "
"They can also be used in vermicomposting with earthworms. "
"This method produces highly nutritious fertilizer for plants."
),

"orange": (
"Food Waste",
"Orange peels can be dried and added to compost bins. "
"They can also be used to create natural cleaning enzymes. "
"These peels contain oils that help remove grease and odors. "
"Recycling citrus waste reduces kitchen waste and supports sustainability."
),

"broccoli": (
"Food Waste",
"Broccoli stems and scraps can be composted easily. "
"They decompose quickly and improve soil organic matter. "
"This helps increase plant growth and soil fertility. "
"Composting vegetable waste reduces landfill garbage."
),

"book": (
"Paper Waste",
"Old books can be donated to libraries, schools, or community centers. "
"If damaged, they can be recycled as paper waste. "
"Recycling paper reduces deforestation and saves natural resources. "
"Paper recycling also uses less energy than producing new paper."
),

"newspaper": (
"Paper Waste",
"Newspapers can be reused for packing or cleaning purposes. "
"They can also be recycled to produce new paper products. "
"Recycling paper reduces the number of trees cut down. "
"It also helps reduce landfill waste and pollution."
),

"can": (
"Scrap Metal",
"Metal cans should be cleaned before recycling. "
"They can be melted and reused to create new metal products. "
"Recycling metal saves energy and reduces mining activities. "
"It also decreases environmental pollution."
),

"knife": (
"Metal Waste",
"Old metal tools can often be repaired or sharpened for reuse. "
"If damaged, they should be sent to metal recycling centers. "
"Metal recycling conserves natural resources. "
"It also reduces the need for new metal production."
),

"spoon": (
"Metal Waste",
"Metal spoons can be reused or donated if still usable. "
"Damaged spoons can be melted and recycled into new products. "
"Metal recycling reduces environmental damage from mining. "
"It also supports sustainable manufacturing."
),

"laptop": (
"E-Waste",
"Old laptops should be sent to certified e-waste recycling facilities. "
"Electronic components contain valuable metals that can be recovered. "
"Proper recycling prevents toxic chemicals from entering the environment. "
"It also helps recover reusable materials."
),

"cell phone": (
"E-Waste",
"Old mobile phones should be recycled through e-waste programs. "
"Many parts such as batteries and circuits can be reused. "
"Proper disposal prevents harmful chemicals from polluting soil and water. "
"It also allows recovery of valuable metals."
),

"tv": (
"E-Waste",
"Televisions contain electronic components that must be recycled responsibly. "
"They should be taken to authorized e-waste collection centers. "
"Improper disposal can release toxic substances. "
"Recycling helps recover metals and reduce environmental damage."
),

"plastic bag": (
"Plastic Waste",
"Plastic bags should be reused whenever possible. "
"They can also be repurposed for trash liners or storage. "
"If damaged, send them to plastic recycling centers. "
"Reducing plastic bag usage helps protect wildlife and oceans."
),

"cardboard": (
"Paper Waste",
"Cardboard boxes can be reused for storage or packaging. "
"They can also be recycled into new paper products. "
"Recycling cardboard saves trees and reduces landfill waste. "
"It also uses less energy than producing new cardboard."
),

"glass bottle": (
"Glass Waste",
"Glass bottles can be washed and reused many times. "
"If broken, they should be sent to glass recycling facilities. "
"Recycled glass can be melted to create new containers. "
"This process saves raw materials and energy."
),

"plastic container": (
"Plastic Waste",
"Plastic containers can be reused for storing food or small items. "
"If they become damaged, they should be sent for plastic recycling. "
"Recycling plastic reduces environmental pollution. "
"It also helps conserve natural resources."
)

}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    results = model(filepath)

    detected_objects = []

    for result in results:
        names = result.names
        for cls in result.boxes.cls:

            label = names[int(cls)]

            category, suggestion = tips.get(
                label,
                (
                    "General Waste",
                    "This item should be separated and disposed responsibly. "
                    "Check if it can be reused or donated before discarding. "
                    "If recyclable, send it to the appropriate recycling facility. "
                    "Responsible waste management helps protect the environment."
                )
            )

            detected_objects.append({
                "name": label,
                "category": category,
                "suggestion": suggestion
            })

    if len(detected_objects) == 0:
        detected_objects.append({
            "name": "No object detected",
            "category": "Unknown",
            "suggestion": "Please upload a clearer image containing visible waste items for better analysis."
        })

    return render_template(
        "index.html",
        filename=file.filename,
        objects=detected_objects
    )


if __name__ == "__main__":
    app.run(debug=True)

    