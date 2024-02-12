import awkward as ak
import hist
from hist.dask import Hist
import gzip
import json
import dask
from distributed import Client
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from coffea.dataset_tools import apply_to_fileset, preprocess

fileset = {
    'DoubleMuon': {
        'files': {
            'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root': {"object_path": "Events"},
            'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root': {"object_path": "Events"},
        }
    },
    'ZZ to 4mu': {
        'files': {
            'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/ZZTo4mu.root': {"object_path": "Events"}
        }
    }
}

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        dataset = events.metadata['dataset']
        muons = ak.zip(
            {
                "pt": events.Muon_pt,
                "eta": events.Muon_eta,
                "phi": events.Muon_phi,
                "mass": events.Muon_mass,
                "charge": events.Muon_charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        h_mass = (
            Hist.new
            .StrCat(["opposite", "same"], name="sign")
            .Log(1000, 0.2, 200., name="mass", label="$m_{\mu\mu}$ [GeV]")
            .Int64()
        )

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) == 0)
        # add first and second muon in every event together
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="opposite", mass=dimuon.mass)

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) != 0)
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="same", mass=dimuon.mass)

        return {
                "entries": ak.num(events, axis=0),
                "mass": h_mass,
        }

    def postprocess(self, accumulator):
        pass

def test_processor_dimu_mass():
    
    # with Client() as _:
    available_fileset, updated_fileset = preprocess(fileset, step_size=2500000, skip_bad_files=True)

    #apply_to_fileset introduces the dataset key to results dictionary
    computable = apply_to_fileset(
        MyProcessor(),
        # available_fileset,
        fileset,
        schemaclass=BaseSchema,
    )
    
    import uproot
    for dataset, dc in fileset.items():
        print(dataset)
        for f in dc['files'].keys():
            of = uproot.open(f)["Events"]
            print(of.keys())
            
    # out, = dask.compute(computable, scheduler="sync")
    # assert out["DoubleMuon"]["entries"] == 1000560
